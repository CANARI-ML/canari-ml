from collections import defaultdict

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torchmetrics import MetricCollection


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: callable,
        learning_rate: float,
        metrics: object,
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

    def on_save_checkpoint(self, checkpoint):
        # Add name of class and path to the lightning module to checkpoint
        # TODO: Add code version/git commit tag to it as well
        checkpoint["lightning_module_name"] = self.__class__.__name__
        checkpoint["lightning_module_path"] = self.__module__


class LitUNet(BaseLightningModule):
    """
    A LightningModule wrapping the UNet implementation of IceNet.
    """

    def __init__(self, *args, **kwargs):
        """
        Construct a UNet LightningModule.
        Note that we keep hyperparameters separate from dataloaders to prevent data leakage at test time.
        :param model: PyTorch model
        :param criterion: PyTorch loss function for training instantiated with reduction="none"
        :param learning_rate: Float learning rate for our optimiser
        """
        super().__init__(*args, **kwargs)

        self.metrics_history = defaultdict(list)

        train_metrics = {}
        val_metrics = {}
        test_metrics = {}

        for metric in self.metrics:
            metric_name = metric.__name__.lower()

            # Overall metrics
            train_metrics.update({f"{metric_name}": metric()})
            val_metrics.update({f"val_{metric_name}": metric()})
            test_metrics.update({f"test_{metric_name}": metric()})

            # Metrics across each leadtime
            if self.enable_leadtime_metrics:
                for i in range(self.model.lead_time):
                    val_metrics.update(
                        {f"val_{metric_name}_{i}": metric(leadtimes_to_evaluate=[i])}
                    )
                    test_metrics.update(
                        {f"test_{metric_name}_{i}": metric(leadtimes_to_evaluate=[i])}
                    )

        self.train_metrics = MetricCollection(train_metrics)
        self.val_metrics = MetricCollection(val_metrics)
        self.test_metrics = MetricCollection(test_metrics)

    def forward(self, x):
        """
        Implement forward function.
        :param x: Inputs to model.
        :return: Outputs of model.
        """
        return self.model(x)

    def training_step(self, batch):
        """
        Perform a pass through a batch of training data.
        Apply pixel-weighted loss by manually reducing.
        See e.g. https://discuss.pytorch.org/t/unet-pixel-wise-weighted-loss-function/46689/5.
        :param batch: Batch of input, output, weight triplets
        :param batch_idx: Index of batch
        :return: Loss from this batch of data for use in backprop
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
        self.train_metrics(
            y_hat.squeeze(dim=-2), y.squeeze(dim=-1), sample_weight.squeeze(dim=-1)
        )
        self.log_dict(
            self.train_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": loss}

    def validation_step(self, batch):
        # x: (b, h, w, channels), y: (b, h, w, lead_time, classes), sample_weight: (b, h, w, lead_time, classes)
        x, y, sample_weight = batch["x"], batch["y"], batch["sample_weights"]
        outputs = self.model(x)

        # y_hat: (b, h, w, classes, lead_time)
        y_hat = outputs

        loss = self.criterion(outputs, y, sample_weight)

        self.val_metrics.update(
            y_hat.squeeze(dim=-2), y.squeeze(dim=-1), sample_weight.squeeze(dim=-1)
        )

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )  # epoch-level loss
        self.log_dict(
            self.val_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )  # epoch-level metrics
        return {"val_loss", loss}

    def test_step(self, batch):
        x, y, sample_weight = batch["x"], batch["y"], batch["sample_weights"]
        outputs = self.model(x)
        y_hat = outputs

        loss = self.criterion(outputs, y, sample_weight)

        self.test_metrics.update(
            y_hat.squeeze(dim=-2), y.squeeze(dim=-1), sample_weight.squeeze(dim=-1)
        )

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )  # epoch-level loss
        return loss

    def on_train_epoch_end(self):
        """
        Reference lightning v2.0.0 migration guide:
        https://github.com/Lightning-AI/pytorch-lightning/pull/16520
        """
        # Reference: https://github.com/Lightning-AI/pytorch-lightning/issues/13147#issuecomment-1138975446
        avg_train_loss = self.trainer.callback_metrics["loss"]
        self.metrics_history["loss"].append(avg_train_loss.item())

        # Reset metrics computed in each training step
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        """
        Reference lightning v2.0.0 migration guide:
        https://github.com/Lightning-AI/pytorch-lightning/pull/16520
        """
        avg_val_loss = self.trainer.callback_metrics["val_loss"]
        self.metrics_history["val_loss"].append(avg_val_loss.item())

        val_metrics = self.val_metrics.compute()
        # self.log_dict(val_metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)  # epoch-level metrics

        for metric in val_metrics:
            self.metrics_history[metric].append(val_metrics[metric].item())

        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(
            self.test_metrics.compute(), on_step=False, on_epoch=True, sync_dist=True
        )  # epoch-level metrics
        self.test_metrics.reset()

    def predict_step(self, batch):
        """
        :param batch: Batch of input, output, weight triplets
        :param batch_idx: Index of batch
        :return: Predictions for given input.
        """
        x, y, sample_weight = batch
        y_hat = self.model(x)

        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
