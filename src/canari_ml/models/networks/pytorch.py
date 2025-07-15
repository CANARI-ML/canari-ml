"""Main module."""

import logging
import os

import hydra
import lightning.pytorch as pl
import pandas as pd
import orjson
import torch
from icenet.model.networks.base import BaseNetwork
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler

from canari_ml.cli.utils import dynamic_import
from canari_ml.data.dataloader import CANARIMLDataSetTorch

from ...lightning.checkpoints import ModelCheckpointOnImprovement


class PytorchNetwork(BaseNetwork):
    def __init__(
        self,
        *args,
        checkpoint_mode: str = "min",
        checkpoint_monitor: str = None,
        early_stopping_patience: int = 0,
        lr_decay: tuple = (1.0, 0, 0),
        pre_load_path: str = None,
        tensorboard_logdir: str = None,
        verbose: bool = False,
        **kwargs,
    ):
        self._checkpoint_mode = checkpoint_mode
        self._checkpoint_monitor = checkpoint_monitor
        self._early_stopping_patience = early_stopping_patience
        self._lr_decay = lr_decay
        self._tensorboard_logdir = tensorboard_logdir

        super().__init__(*args, **kwargs)

        self._weights_path = os.path.join(
            self.network_folder,
            "{}.network_{}.{}.h5".format(
                self.run_name, self.dataset.identifier, self.seed
            ),
        )

        if pre_load_path is not None and not os.path.exists(pre_load_path):
            raise RuntimeError(
                "{} is not available, so you cannot preload the "
                "network with it!".format(pre_load_path)
            )
        self._pre_load_path = pre_load_path

        self._verbose = verbose

        torch.set_float32_matmul_precision("medium")

    def _attempt_seed_setup(self):
        super()._attempt_seed_setup()
        pl.seed_everything(self._seed)

    def train(
        self,
        epochs: int,
        model_creator: callable,
        train_dataloader: object,
        model_creator_kwargs: dict = None,
        save: bool = True,
        validation_dataloader: object = None,
    ):
        history_path = os.path.join(
            self.network_folder, "{}_{}_history.json".format(self.run_name, self.seed)
        )

        logger_name = f"{self.run_name}_{self.seed}"

        lit_module = model_creator(**model_creator_kwargs)

        tb_dir = "tb_logs"
        logger = TensorBoardLogger(tb_dir, name=logger_name)
        # logger = CSVLogger("logs", name=logger_name)    # Uses basic CSV logging

        # Print model summary
        print(lit_module.model)

        profiler = PyTorchProfiler(
            on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_dir)
        )

        # precision options:
        # ('transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed',
        # 'bf16-true', 'bf16-mixed', '32-true', '64-true', 64, 32, 16, '64', '32',
        # '16', 'bf16')
        # set up trainer configuration
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            # strategy="ddp",
            precision="16-mixed",  # Enable 16-bit precision (mixed precision training)
            # precision="16-true",  # Enable true 16-bit precision
            # precision="bf16-true",  # Enable true bfloat 16-bit precision
            log_every_n_steps=5,
            max_epochs=epochs,
            # auto_lr_find=True,
            num_sanity_val_steps=0,
            enable_checkpointing=False,  # Disable built-in checkpointing, using callback instead
            logger=logger,
            deterministic=True,
            # benchmark=True,
            # profiler=profiler,
            # enable_progress_bar=False,
            callbacks=[
                # RichProgressBar(leave=True),
                # TQDMProgressBar(leave=True),
            ],
            fast_dev_run=False,  # Runs single batch through train and validation
            #    when running trainer.test()
            # Note: Cannot use with automatic best checkpointing
        )
        # trainer.tune(lit_module.model)
        # # Run learning rate finder
        # lr_finder = trainer.tuner.lr_find(lit_module.model)

        # # Results can be found in
        # lr_finder.results

        # # Plot with
        # fig = lr_finder.plot(suggest=True)
        # fig.show()
        # exit()

        if save:
            # checkpoint_weights_filename = (
            #     "checkpoint.{}.network_{}.{}.".format(
            #         self.run_name, self.dataset.identifier, self.seed
            #     )
            #     + "{epoch:03d}"
            # )

            # # Save weights each time monitored metric has improved (creates new file each time)
            # checkpoint_weights_callback = ModelCheckpointOnImprovement(
            #     monitor=self._checkpoint_monitor,
            #     mode=self._checkpoint_mode,
            #     save_top_k=-1,
            #     # every_n_epochs=1,
            #     enable_version_counter=False,
            #     filename=checkpoint_weights_filename,
            #     # Prevents "epoch=001" in filename output
            #     auto_insert_metric_name=False,
            #     # dirpath=self._weights_path,
            #     dirpath=self.network_folder,
            #     save_weights_only=True,
            # )

            # logging.info(
            #     "Saving network to: {}/{}.ckpt".format(
            #         self.network_folder, checkpoint_weights_filename
            #     )
            # )
            # trainer.callbacks.append(checkpoint_weights_callback)

            # checkpoint_model_filename = "{}.model_{}.{}".format(
            #     self.run_name, self.dataset.identifier, self.seed
            # )
            checkpoint_model_filename = "{}.model_{}.{}".format(
                self.run_name, self.dataset.identifier, self.seed
            ) #+ ".epoch{epoch:02d}"

            # Save entire model including weights of the best monitored epoch (overwrites previous best)
            checkpoint_model_callback = ModelCheckpoint(
                monitor=self._checkpoint_monitor,
                mode=self._checkpoint_mode,
                save_top_k=1,
                # every_n_epochs=1,
                enable_version_counter=False,
                # save_on_train_epoch_end=False,
                filename=checkpoint_model_filename,
                # Prevents "epoch=001" in filename output
                auto_insert_metric_name=False,
                dirpath=self.network_folder,
                save_weights_only=False,
            )

            logging.info("Saving model to: {}".format(checkpoint_model_filename))
            trainer.callbacks.append(checkpoint_model_callback)

        # print(f"Training {len(train_dataset)} examples / {len(train_dataloader)} batches (batch size {batch_size}).")
        # print(f"Validating {len(validation_dataset)} examples / {len(val_dataloader)} batches (batch size {batch_size}).")
        if self._pre_load_path and os.path.exists(self._pre_load_path):
            logging.warning(
                "Automagically loading network weights from {}".format(
                    self._pre_load_path
                )
            )

        # train model
        trainer.fit(
            lit_module,
            train_dataloader,
            validation_dataloader,
            ckpt_path=None,
        )

        with open(history_path, "w") as fh:
            logging.info(f"Saving metrics history to: {history_path}")
            pd.DataFrame(lit_module.metrics_history).to_json(fh)

        # # TODO: consider using .keras format throughout
        # # TODO: need to consider pre_load / create and save functionality for checkpoint recovery
        # if self._pre_load_path and os.path.exists(self._pre_load_path):
        #     logging.warning("Automagically loading network weights from {}".format(
        #         self._pre_load_path))
        #     network.load_weights(self._pre_load_path)

        # network.summary()

        # if save:
        #     logging.info("Saving network to: {}".format(self._weights_path))
        #     network.save_weights(self._weights_path)
        #     logging.info("Saving model to: {}".format(self.model_path))
        #     save_model(network, self.model_path)
        ## To save model history, should define a callback to process the logging output.
        #     with open(history_path, 'w') as fh:
        #         pd.DataFrame(model_history.history).to_json(fh)

        return trainer, checkpoint_model_callback


class HYDRAPytorchNetwork(BaseNetwork):
    def __init__(
        self,
        cfg,
        *args,
        tensorboard_logdir: str | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        self._cfg = cfg
        verbose=cfg.train.verbose
        self._output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        logging.info(f"Working directory: {self._output_dir}")

        # Get directory where cached data is stored for training
        with open(cfg.train.dataset) as f:
            dataset_json = f.read()
        parsed_json = orjson.loads(dataset_json)
        dataset_identifier = parsed_json["identifier"]
        network_folder = os.path.join(
            os.path.dirname(cfg.train.dataset), dataset_identifier
        )

        dataset = CANARIMLDataSetTorch(
            configuration_path=cfg.train.dataset,
            batch_size=cfg.train.batch_size,
            shuffling=cfg.train.shuffling,
            path=os.path.dirname(cfg.train.dataset),
        )

        super_kwargs = {
            "dataset": dataset,
            "run_name": cfg.train.run_name,
            "network_folder": network_folder,
            "seed": cfg.train.seed,
        }

        self._tensorboard_logdir = tensorboard_logdir

        super().__init__(**super_kwargs)

        # if pre_load_path is not None and not os.path.exists(pre_load_path):
        #     raise RuntimeError(
        #         "{} is not available, so you cannot preload the "
        #         "network with it!".format(pre_load_path)
        #     )
        # self._pre_load_path = pre_load_path

        self._verbose = verbose

        torch.set_float32_matmul_precision("medium")

    def _attempt_seed_setup(self):
        super()._attempt_seed_setup()
        pl.seed_everything(self._seed)

    def train(
        self,
        # epochs: int,
        # lit_module,
        # dataset,
        # train_dataloader: object,
        # validation_dataloader: object = None,
        # save: bool = True,
    ):
        # Module initialisation
        cfg = self._cfg # HYDRA loaded configuration
        dataset = self._dataset
        lead_time = dataset.lead_time
        input_shape = (dataset.num_channels, *dataset.shape) # channels, height, width
        train_dataloader, validation_dataloader, _ = dataset.get_data_loaders(ratio=1.0)

        network_partial = hydra.utils.instantiate(cfg.model.network)
        network = network_partial(input_channels=input_shape[0], lead_time=lead_time)

        litmodule_partial = hydra.utils.instantiate(cfg.model.litmodule)
        metric_import_paths = cfg.model.litmodule.metrics
        metrics = [dynamic_import(path) for path in metric_import_paths]

        litmodule = litmodule_partial(model=network, metrics=metrics)

        # Output initialisation
        history_path = os.path.join(
            self._output_dir, "{}_{}_history.json".format(self.run_name, self.seed)
        )

        logger_name = f"{self.run_name}_{self.seed}"

        tb_dir = "tb_logs"
        logger = TensorBoardLogger(tb_dir, name=logger_name)
        # logger = CSVLogger("logs", name=logger_name)    # Uses basic CSV logging

        # Print model summary
        print(litmodule.model)

        profiler = PyTorchProfiler(
            on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_dir)
        )

        # Trainer set-up
        # Initialise callback functions from HYDRA configuration
        callbacks = self.add_callback(cfg.callbacks)

        # precision options:
        # ('transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed',
        # 'bf16-true', 'bf16-mixed', '32-true', '64-true', 64, 32, 16, '64', '32',
        # '16', 'bf16')
        # set up trainer configuration
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            # strategy="ddp",
            precision="16-mixed",  # Enable 16-bit precision (mixed precision training)
            # precision="16-true",  # Enable true 16-bit precision
            # precision="bf16-true",  # Enable true bfloat 16-bit precision
            log_every_n_steps=5,
            max_epochs=cfg.train.max_epochs,
            # auto_lr_find=True,
            num_sanity_val_steps=0,
            # enable_checkpointing=False,  # Disable built-in checkpointing, using callback instead
            logger=logger,
            deterministic=True,
            # benchmark=True,
            # profiler=profiler,
            # enable_progress_bar=False,
            # callbacks=[],
            callbacks=callbacks,
            # callbacks=self._callbacks,
            # callbacks=[
            #     # RichProgressBar(leave=True),
            #     # TQDMProgressBar(leave=True),
            # ],
            fast_dev_run=False,  # Runs single batch through train and validation
            #    when running trainer.test()
            # Note: Cannot use with automatic best checkpointing
        )

        # Run training
        trainer.fit(
            litmodule,
            train_dataloader,
            validation_dataloader,
            ckpt_path=None,
        )

        # Save history of metrics
        with open(history_path, "w") as fh:
            logging.info(f"Saving metrics history to: {history_path}")
            pd.DataFrame(litmodule.metrics_history).to_json(fh)

        return trainer
