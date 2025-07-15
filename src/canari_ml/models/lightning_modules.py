from collections import defaultdict
from typing import Iterable

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics import MetricCollection


class BaseLightningModule(pl.LightningModule):
    """
    Base class for all Canari ML models using PyTorch Lightning.

    This module inherits from `pytorch_lightning.LightningModule` and provides the basic
    functionality required for training, validating, and testing Canari ML models. It
    also includes support for saving hyperparameters to checkpoints and recording
    metrics during training and validation.

    Attributes:
        model (nn.Module): The PyTorch model being wrapped.
        criterion (callable): The loss function used during training and validation.
        learning_rate (float): The learning rate used for optimisation.
        metrics (Iterable[callable]): An iterable of callable objects representing the
            metrics to be recorded during training and validation.
        enable_leadtime_metrics (bool, optional): Flag indicating whether to enable
            lead-time related metrics. Defaults to True.
        n_output_classes (int): The number of output classes in the model.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: callable,
        learning_rate: float,
        metrics: Iterable[callable],
        enable_leadtime_metrics: bool = True,
    ):
        super().__init__()
        # Save input parameters to __init__ (hyperparams) when checkpointing.
        # self.save_hyperparameters(ignore=["model", "criterion"])
        self.save_hyperparameters()

        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.metrics = metrics
        self.enable_leadtime_metrics = enable_leadtime_metrics
        self.n_output_classes = (
            model.n_output_classes
        )  # this should be a property of the network

        self.metrics_history = defaultdict(list)

        metrics = {}

        for metric in self.metrics:
            metric_name = metric.__name__.lower()

            # Overall metrics
            metrics.update({f"{metric_name}": metric()})

            # Metrics across each leadtime
            if self.enable_leadtime_metrics:
                for i in range(self.model.lead_time):
                    metrics.update(
                        {f"{metric_name}_{i}": metric(leadtimes_to_evaluate=[i])}
                    )

        metric_collection = MetricCollection(metrics)
        self.train_metrics = metric_collection.clone(prefix="train_")
        self.val_metrics = metric_collection.clone(prefix="val_")
        self.test_metrics = metric_collection.clone(prefix="test_")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement forward function.

        Applies the model to the input tensor `x`.

        Args:
            x: Inputs to the model. Expected shape is
                `(batch_size, channels, height, width)`.

        Returns:
            Output of the model with shape `(batch_size, num_classes, height, width)`.
        """
        return self.model(x)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """
        Override PyTorch Lightning's default `on_save_checkpoint` method to add custom data.

        This method adds the name of the class and the path to the Lightning module to
        the checkpoint.

        Args:
            checkpoint: The checkpoint dictionary to which additional data will be added.
        """
        # Add name of class and path to the lightning module to checkpoint
        # TODO: Add code version/git commit tag to it as well
        checkpoint["lightning_module_name"] = self.__class__.__name__
        checkpoint["lightning_module_path"] = self.__module__


class LitUNet(BaseLightningModule):
    r"""
    A LightningModule wrapping the :class:`UNet` implementation of IceNet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        r"""
        Perform a pass through a batch of training data.

        This method implements the core training loop for a single batch of data.
        It takes the input, output, and sample weights from the provided batch,
        passes the inputs through the model to obtain predictions, computes
        the pixel-weighted loss using the provided criterion, and updates any
        relevant metrics. The computed loss is then returned for use in backpropagation.

        Args:
            batch: A dictionary containing 'x', 'y', and 'sample_weights'
                keys with their respective values representing input data,
                target output data, and sample weights.
            batch_idx: Index of the current batch.

        Returns:
            A dictionary containing the computed loss for this batch
                of data. This is used in backpropagation to update the model's
                parameters.

        .. note::
            The method uses pixel-weighted loss by manually reducing it, following the
            approach outlined
            `here <https://discuss.pytorch.org/t/unet-pixel-wise-weighted-loss-function/46689/5>`_.

            It also logs the computed loss and metrics for use in monitoring training
            progress.
        """
        x, y, sample_weight = batch["x"], batch["y"], batch["sample_weights"]
        outputs = self.model(x)

        loss = self.criterion(outputs, y, sample_weight)

        # This logged result can be accessed later via `self.trainer.callback_metrics("loss")`
        # Reference: https://github.com/Lightning-AI/pytorch-lightning/issues/13147#issuecomment-1138975446
        self.log(
            "loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        # Compute metrics
        y_hat = outputs
        self.train_metrics.update(
            y_hat, y, sample_weight
        )
        self.log_dict(
            self.train_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        return {"loss": loss}

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        r"""
        Perform a pass through a batch of validation data.

        This method implements the core validation loop for a single batch of data.
        The methodology is the same as `training_step`. The computed loss is logged
        for use in monitoring validation progress.

        Args:
            batch: A dictionary containing 'x', 'y', and 'sample_weights'
                keys with their respective values representing input data,
                target output data, and sample weights.
            batch_idx: Index of the current batch.

        Returns:
            dict: A dictionary containing the computed loss for this batch
                of data. This is used in logging to monitor validation progress.
        """
        # x: (b, h, w, channels), y: (b, h, w, lead_time, classes), sample_weight: (b, h, w, lead_time, classes)
        x, y, sample_weight = batch["x"], batch["y"], batch["sample_weights"]
        outputs = self.model(x)

        # y_hat: (b, h, w, classes, lead_time)
        y_hat = outputs

        loss = self.criterion(outputs, y, sample_weight)

        self.val_metrics.update(
            y_hat, y, sample_weight
        )

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )  # epoch-level loss

        self.log_dict(
            self.val_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )  # epoch-level metrics
        return {"val_loss": loss}

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        r"""
        Perform a pass through a batch of test data.

        This method implements the core testing loop for a single batch of data.
        The methodology is the same as `training_step`. The computed loss is logged
        for use in monitoring test progress.

        Args:
            batch: A dictionary containing 'x', 'y', and 'sample_weights'
                keys with their respective values representing input data,
                target output data, and sample weights.
            batch_idx: Index of the current batch.

        Returns:
            A dictionary containing the computed loss for this batch
                of data. This is used in logging to monitor test progress.
        """
        x, y, sample_weight = batch["x"], batch["y"], batch["sample_weights"]
        outputs = self.model(x)
        y_hat = outputs

        loss = self.criterion(outputs, y, sample_weight)

        self.test_metrics.update(
            y_hat, y, sample_weight
        )

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )  # epoch-level loss

        self.log_dict(
            self.test_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )  # epoch-level metrics
        return {"test_loss": loss}

    def on_train_epoch_end(self) -> None:
        r"""
        Perform actions at the end of each training epoch.

        This method is called by PyTorch Lightning at the end of each training epoch.
        It computes and stores the average loss for the completed epoch, then resets
        the metrics computed during individual training steps in preparation for the
        next epoch.

        .. note::
            The implementation follows the migration guide from Lightning v1.5 to v2.0,
            as outlined `here <https://github.com/Lightning-AI/pytorch-lightning/pull/16520>`_.

            It also references `this issue <https://github.com/Lightning-AI/pytorch-lightning/issues/13147#issuecomment-1138975446>`_
            for accessing logged results.
        """
        # Reference: https://github.com/Lightning-AI/pytorch-lightning/issues/13147#issuecomment-1138975446
        avg_train_loss = self.trainer.callback_metrics["loss"]
        self.metrics_history["loss"].append(avg_train_loss.item())

        # Reset metrics computed in each training step
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        r"""
        Perform actions at the end of each validation epoch.

        This method is called by PyTorch Lightning at the end of each validation epoch.
        It computes and stores the average loss for the completed epoch, then updates,
        stores, and resets the metrics computed during individual validation steps in
        preparation for the next epoch.
        """
        avg_val_loss = self.trainer.callback_metrics["val_loss"]
        self.metrics_history["val_loss"].append(avg_val_loss.item())

        val_metrics = self.val_metrics.compute()
        # self.log_dict(val_metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)  # epoch-level metrics

        for metric in val_metrics:
            self.metrics_history[metric].append(val_metrics[metric].item())

        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """
        Perform actions at the end of each test epoch.

        This method is called by PyTorch Lightning at the end of each test epoch.
        It logs and resets the metrics computed during individual test steps in
        preparation for the next epoch.
        """
        self.log_dict(
            self.test_metrics.compute(), on_step=False, on_epoch=True, sync_dist=True
        )  # epoch-level metrics
        self.test_metrics.reset()

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Generate predictions for a given input batch.

        This method is called by PyTorch Lightning during prediction to generate model
        outputs for the provided input batch. It returns the model's predictions based
        on the input data.

        Args:
            batch: A dictionary containing the input ('x') and output ('y')
                tensors, as well as any additional relevant information like
                'sample_weights'.
            batch_idx: The index of the current batch in the dataloader.

        Returns:
            y_hat: The model's predictions for the given input batch.
        """
        x, y, sample_weight = batch["x"], batch["y"], batch["sample_weights"]
        y_hat = self.model(x)

        return y_hat

    def configure_optimizers(self) -> dict:
        r"""
        Configure and return the optimizer and learning rate scheduler.

        This method is called by PyTorch Lightning to initialise the optimizer and
        learning rate scheduler used for training the model. It returns a dictionary
        containing both the optimizer and the lr_scheduler.

        Returns:
            A dictionary containing the optimizer and lr_scheduler.
                - optimizer (torch.optim.optimizer.Optimizer): The optimizer instance
                    used to update model parameters during training.
                - lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning
                    rate scheduler instance, which adjusts the learning rate over time
                    based on specified criteria.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-5
        )
        # Ref OneCycleLR: https://medium.com/@g.martino8/one-cycle-lr-scheduler-a-simple-guide-c3aa9c4cbd9f
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer, max_lr=0.01, steps_per_epoch=64, epochs=10, three_phase=True
        # )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
