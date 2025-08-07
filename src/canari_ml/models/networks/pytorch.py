"""Main module."""

import datetime as dt
import glob
import logging
import os
from abc import abstractmethod
from pathlib import Path

import hydra
import lightning.pytorch as pl
import numpy as np
import orjson
import pandas as pd
import torch

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class
from lightning import Callback
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
from omegaconf import DictConfig

from canari_ml.cli.utils import dynamic_import
from canari_ml.data.dataloader import CANARIMLDataSetTorch
from canari_ml.preprocess.utils import symlink

from ...lightning.checkpoints import ModelCheckpointOnImprovement

CACHE_SYMLINK_DIR = "cache_dir"

class BaseNetwork:
    """
    Base class for managing network training, prediction, and callback handling.

    This class is a parent class for creating, training, and evaluating neural networks.
    It manages the model folder structure, seed setup for reproducibility, and handles
    default callbacks. Subclasses must implement the `train` and `predict` methods.

    Attributes:
        _network_folder: Path to the directory where network outputs are stored.
        _dataset: The dataset used for training/prediction.
        _callbacks: List of callback objects for monitoring/training procedures.
    """
    def __init__(self,
                 dataset: CANARIMLDataSetTorch,
                 run_name: object,
                 callbacks_additional: list | None = None,
                 callbacks_default: list | None = None,
                 network_folder: object | None = None,
                 seed: int = 42):
        """
        Initialise the BaseNetwork instance.

        Args:
            dataset: The dataset to be used for training/prediction.
            run_name: Identifier for the current run, used in folder naming.
            callbacks_additional: List of additional callback objects to add.
            callbacks_default: List of default callbacks (if not using defaults).
            network_folder: Custom path for the network output directory.
            seed: Random seed for reproducibility.
        """

        if network_folder:
            self._network_folder = network_folder
            # self._network_folder = os.path.join(".", "results", "networks", run_name)

            if not os.path.exists(self._network_folder):
                logging.info("Creating network folder: {}".format(network_folder))
                os.makedirs(self._network_folder, exist_ok=True)

        self._dataset = dataset
        self._run_name = run_name
        self._seed = seed
        self._output_dir: str = ""

        self._callbacks = (
            self.get_default_callbacks()
            if callbacks_default is None
            else callbacks_default
        )
        self._callbacks += (
            callbacks_additional if callbacks_additional is not None else []
        )

        self._attempt_seed_setup()

    def _attempt_seed_setup(self):
        logging.warning(
            "Setting seed for best attempt at determinism, value {}".format(self._seed))
        # determinism is not guaranteed across different versions of PyTorch.
        # determinism is not guaranteed across different hardware.
        os.environ['PYTHONHASHSEED'] = str(self._seed)
        pl.seed_everything(self._seed)

    def add_callback(self, callback: Callback | DictConfig) -> list[Callback]:
        if isinstance(callback, DictConfig):
            for cb_name, cb in callback.items():
                if "_target_" in cb:
                    logging.info("Adding callback: {}".format(cb._target_))
                    self._callbacks.append(hydra.utils.instantiate(cb))
        else:
            logging.info("Adding callback {}".format(callback))
            self._callbacks.append(callback)
        return self._callbacks

    def get_default_callbacks(self):
        return list()

    def create_cache_symlink(self, target_path: str):
        # Create symlink to cache dir output in train/pred output location, e.g.:
        # self._output_dir = outputs/{train_name}/training/42/cache_dir
        # symlink_path = outputs/{train_name}/training/cache_dir
        symlink_path = os.path.join(
            os.path.dirname(self._output_dir), CACHE_SYMLINK_DIR
        )
        if os.path.realpath(target_path) != os.path.realpath(symlink_path):
            symlink_dir = os.path.dirname(symlink_path)
            relative_target = os.path.relpath(target_path, symlink_dir)
            os.symlink(relative_target, symlink_path)

    def save_prediction(
        self, predictions: torch.tensor, dates: list[dt.datetime], output_folder: str
    ) -> None:
        """
        Save raw prediction outputs to numpy files.

        Args:
            predictions: Tensor containing model forecasts.
            dates: List of date objects corresponding to predictions.
            output_folder: Directory path where files will be saved.
        """
        if os.path.exists(output_folder):
            logging.warning("{} output already exists".format(output_folder))
        os.makedirs(output_folder, exist_ok=True)

        idx = 0
        for workers, prediction in enumerate(predictions):
            for batch in range(prediction.shape[0]):
                date = dates[idx]
                logging.info(
                    "Saving {} - forecast output {}".format(date, prediction.shape)
                )
                output_path = os.path.join(output_folder, date.strftime("%Y_%m_%d.npy"))
                forecast = prediction[batch, :, :, :, :]
                forecast_np = forecast.detach().cpu().numpy()
                np.save(output_path, forecast_np)
                idx += 1

    @abstractmethod
    def train(self,
              epochs: int,
              model_creator: callable,
              train_dataset: object,
              model_creator_kwargs: dict = None,
              save: bool = True) -> pl.Trainer:
        """
        Train the neural network.

        Must be implemented by subclasses.

        Args:
            epochs: Number of training epochs.
            model_creator: Callable to instantiate the model.
            train_dataset: Dataset for training.
            model_creator_kwargs: Keyword arguments for model creation.
            save: Whether to save the trained model.

        Raises:
            NotImplementedError: If not overridden in subclass.
        """
        raise NotImplementedError("Implementation not found")

    @abstractmethod
    def predict(self) -> None:
        """
        Evaluate a pre-trained neural network.

        Must be implemented by subclasses.

        Raises:
            NotImplementedError: If not overridden in subclass.
        """
        raise NotImplementedError("Implementation not found")

    @property
    def callbacks(self):
        return self._callbacks

    @property
    def dataset(self):
        return self._dataset

    @property
    def network_folder(self):
        return self._network_folder

    @property
    def run_name(self):
        return self._run_name

    @property
    def seed(self):
        return self._seed


class HYDRAPytorchNetwork(BaseNetwork):
    def __init__(
        self,
        cfg,
        *args,
        run_type: str = "train",
        verbose: bool = False,
        **kwargs,
    ):
        self._cfg = cfg
        verbose=cfg.verbose

        if run_type == "train":
            cache_path = os.path.dirname(cfg.train.dataset)
            self._train_cache_path = cache_path
            # Get directory where cached data is stored for training
            with open(cfg.train.dataset) as f:
                dataset_json = f.read()
            parsed_json = orjson.loads(dataset_json)
            dataset_identifier = parsed_json["identifier"]
            # Path to cached data output for training
            network_folder = os.path.join(
                cache_path, dataset_identifier
            )
            dataset = CANARIMLDataSetTorch(
                configuration_path=cfg.train.dataset,
                batch_size=cfg.train.batch_size,
                shuffling=cfg.train.shuffling,
                path=cache_path,
            )
            seed = cfg.train.seed
        elif run_type == "predict":
            cache_path = os.path.dirname(cfg.predict.dataset)
            self._predict_cache_path = cache_path
            network_folder = None
            dataset = CANARIMLDataSetTorch(
                configuration_path=cfg.predict.dataset,
                batch_size=cfg.predict.batch_size,
                shuffling=False,
                path=cache_path,
            )
            seed = cfg.predict.seed
        run_name = cfg.train.name

        super_kwargs = {
            "dataset": dataset,
            "run_name": run_name,
            "network_folder": network_folder,
            "seed": seed,
        }

        super().__init__(**super_kwargs)

        self._output_dir = HydraConfig.get().runtime.output_dir
        logging.info(f"Working directory: {self._output_dir}")

        self._verbose = verbose

        torch.set_float32_matmul_precision("medium")

    def train(self) -> pl.Trainer:
        # Module initialisation
        cfg = self._cfg # HYDRA loaded configuration
        dataset = self._dataset
        lead_time = dataset.lead_time
        input_shape = (dataset.num_channels, *dataset.shape) # channels, height, width
        train_dataloader, validation_dataloader, _ = dataset.get_data_loaders(ratio=1.0)

        # Initialise neural network
        network_partial = hydra.utils.instantiate(cfg.model.network)
        network = network_partial(input_channels=input_shape[0], lead_time=lead_time)

        # Initialise Lightning module
        litmodule_partial = hydra.utils.instantiate(cfg.model.litmodule)
        metric_import_paths = cfg.model.litmodule.metrics
        metrics = [dynamic_import(path) for path in metric_import_paths]

        litmodule = litmodule_partial(model=network, metrics=metrics)

        # Output initialisation
        history_path = os.path.join(
            self._output_dir, "{}_{}_history.json".format(self.run_name, self.seed)
        )

        # Print model summary
        print(litmodule.model)

        # # Finish prior running wandb instances (if any)
        # if wandb.run is not None:
        #     wandb.finish()

        # Initialise logger
        logger = hydra.utils.instantiate(cfg.logger)
        wandb_run_id = None
        if isinstance(logger, WandbLogger) and logger.experiment is not None:
            wandb_run_id = logger.experiment.id
            logging.info(f"W&B Run: {logger.experiment.name}")

        # Trainer set-up
        trainer_kwargs = {
            "logger": logger,
            # Initialise callback functions from HYDRA configuration
            "callbacks": self.add_callback(cfg.callbacks),
        }

        # Initialise profiler if enabled
        profiler = hydra.utils.instantiate(cfg.profiler) if "profiler" in cfg else None
        trainer_kwargs.update({"profiler": profiler})

        # Initialise Lightning Trainer
        trainer_partial = hydra.utils.instantiate(cfg.trainer)
        trainer = trainer_partial(**trainer_kwargs)

        # Store seed and wandb experiment id (if enabled) in checkpoint
        hyperparams = {"seed": self.seed}
        if wandb_run_id:
            hyperparams["wandb_run_id"] = wandb_run_id
        trainer.logger.log_hyperparams(hyperparams)

        # Run training
        trainer.fit(
            litmodule,
            train_dataloader,
            validation_dataloader,
            ckpt_path=None, # Outputs to default: HYDRA_OUTPUT_PATH/checkpoints/*.ckpt
        )

        # # Save history of metrics
        # with open(history_path, "w") as fh:
        #     logging.info(f"Saving metrics history to: {history_path}")
        #     pd.DataFrame(litmodule.metrics_history).to_json(fh)

        # Finish running wandb instances (if wandb is being used)
        if isinstance(logger, WandbLogger):
            logger.experiment.finish()
            # logger.finalize(status="success")

        # Create a symlink to the dataset used for this run to output dir
        # Will make further postprocessing much easier for user
        self.create_cache_symlink(target_path=self._train_cache_path)

        return trainer

    def predict(self, test_set: bool = False) -> None:
        # Module initialisation
        cfg = self._cfg # HYDRA loaded configuration

        dataset = self._dataset

        # Use dummy=True to prevent parent `DataCollection` from making
        # dir at base_path
        dl = dataset.get_data_loader(base_path="", dummy=True)

        dataset = self._dataset
        lead_time = dataset.lead_time
        input_shape = (dataset.num_channels, *dataset.shape) # channels, height, width

        # Initialise neural network
        network_partial = hydra.utils.instantiate(cfg.model.network)
        network = network_partial(input_channels=input_shape[0], lead_time=lead_time)

        # Initialise Lightning module
        litmodule_partial = hydra.utils.instantiate(cfg.model.litmodule)
        metric_import_paths = cfg.model.litmodule.metrics
        metrics = [dynamic_import(path) for path in metric_import_paths]

        # Assuming default model checkpoint output location
        checkpoint_path = cfg.predict.checkpoint_path
        train_run_name = cfg.train.name
        seed = str(cfg.predict.seed)
        if not checkpoint_path:
            ckpt_dir = Path("outputs") / train_run_name / "training" / seed / "checkpoints"
            checkpoint_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
            # Remove last.ckpt file from ckpts found (only saved for resuming training)
            last_cpkt = "last.ckpt"
            for i, checkpoint_file in enumerate(checkpoint_files):
                if last_cpkt in checkpoint_file:
                    checkpoint_files.pop(i)
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
            checkpoint_path = checkpoint_files[-1]  # use last (latest) by name
            logging.info(f"Using checkpoint: {checkpoint_path}")

        # Get LightningModule class (not the instance)
        litmodule_class = get_class(cfg.model.litmodule._target_)

        litmodule = litmodule_class.load_from_checkpoint(
            checkpoint_path,
            model=network,
            metrics=metrics,
        )
        litmodule.to("cpu")

        litmodule.eval()

        # Path to output raw numpy predictions to
        output_folder = os.path.join(self._output_dir, "raw_predictions")

        if not test_set:
            logging.info("Generating forecast inputs from processed netCDF files")
            predict_dates = [
                dt.datetime.strptime(date, "%Y-%m-%d")
                for date in cfg.predict.dates
            ]
            for date in predict_dates:
                x, base_ua700, y, sample_weights = dl.generate_sample(date, prediction=True)

                # input_sample shape: (1, channels, height, width)
                input_sample = torch.tensor(x).unsqueeze(dim=0)
                # Expand input_sample to match prediction shape's lead time dimension
                base_ua700_expanded = torch.tensor(base_ua700.data).unsqueeze(dim=-1)

                logging.info("Running prediction {}".format(date))
                with torch.no_grad():
                    predictions = litmodule(input_sample).unsqueeze(dim=0)

                    # Add input state to predicted delta to get absolute forecast
                    absolute_forecast = predictions + base_ua700_expanded

                self.save_prediction(
                    predictions=absolute_forecast,
                    dates=[date],
                    output_folder=output_folder,
                )
        else:
            logging.info("Using test set from cached files")
            # TODO: Need to update this to handle adding `base_ua700` to predictions
            _, _, test_dataloader = dataset.get_data_loaders(ratio=1.0)

            trainer = pl.Trainer()
            with torch.no_grad():
                predictions = trainer.predict(litmodule, dataloaders=test_dataloader)

            source_key = [k for k in dl.config["sources"].keys() if k != "meta"][0]

            test_dates = [
                dt.datetime.strptime(d, "%Y-%m-%d")
                for d in dl.config["sources"][source_key]["splits"]["test"]
            ]

            self.save_prediction(
                predictions=predictions,
                dates=test_dates,
                output_folder=output_folder,
            )

        # Create a symlink to the dataset used for this run to output dir
        # Will make further postprocessing much easier for user
        self.create_cache_symlink(target_path=self._predict_cache_path)

        return
