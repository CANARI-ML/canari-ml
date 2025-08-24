import logging
import os
import sys
from pathlib import Path

import hydra
import orjson
from download_toolbox.cli import csv_of_date_args
from download_toolbox.interface import (
    get_dataset_config_implementation,
    get_implementation,
)
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from preprocess_toolbox.cli import process_split_args
from preprocess_toolbox.dataset.cli import init_dataset
from preprocess_toolbox.utils import update_config

from canari_ml.data.processors.cds import ERA5PreProcessor
from canari_ml.data.masks.era5 import Masks
from canari_ml.preprocess.reproject import reproject_datasets_from_config
from canari_ml.preprocess.utils import (
    compute_loader_hash,
    compute_step_hash,
    symlink,
    IterableNamespace,
)
from canari_ml.models.networks.pytorch import NORMALISATION_SYMLINK_DIR
from canari_ml.data.loaders import CanariMLDataLoaderFactory

logger = logging.getLogger(__name__)


def get_date_splits(cfg: DictConfig) -> IterableNamespace:
    """
    Get the start and end dates for training, validation, and testing splits
    based on the configuration.

    Args:
        cfg: Hydra preprocess configuration object.

    Returns:
        An iterable namespace with split information.
    """
    if cfg.preprocess_type == "train":
        split_starts_dict = {
            "train": "|".join(cfg.input.dates.train.start),
            "val": "|".join(cfg.input.dates.val.start),
            "test": "|".join(cfg.input.dates.test.start),
        }
        split_ends_dict = {
            "train": "|".join(cfg.input.dates.train.end),
            "val": "|".join(cfg.input.dates.val.end),
            "test": "|".join(cfg.input.dates.test.end),
        }
    elif cfg.preprocess_type == "predict":
        split_starts_dict = {
            "predict": "|".join(cfg.input.dates.predict.start),
        }
        split_ends_dict = {
            "train": "|".join(cfg.input.dates.predict.end),
        }
    else:
        raise ValueError(f"Unrecognised `preprocess_type`: {cfg.preprocess_type}")

    split_starts = csv_of_date_args(
        ",".join(split for split in split_starts_dict.values())
    )
    split_ends = csv_of_date_args(",".join(split for split in split_ends_dict.values()))
    additional_args = IterableNamespace(
        splits=cfg.preprocess_type,
        split_names=split_starts_dict.keys(),
        split_starts=split_starts,
        split_ends=split_ends,
    )

    return additional_args


def reproject(cfg: DictConfig) -> None:
    """
    Reproject data using the specified configuration file.

    Args:
        cfg: Hydra preprocess configuration object.
    """

    if Path(cfg.paths.reproject.config_file).exists():
        logger.warning(
            f"(skipping) Reprojection step already completed previously: {cfg.paths.reproject.config_file}"
        )
        return

    # Emulating argparse interface (only for interface with preprocess toolbox)
    main_args = IterableNamespace(
        source=cfg.paths.download.config_file,                  # Input: downloader json config
        destination_id=cfg.source_dataset_id,                   # Input: identifier
        split_head=cfg.input.lag_length,                        # Input: Lag
        split_tail=cfg.input.forecast_length,                   # Input: Leadtime
        workers=cfg.workers,                                    # Input: Concurrent workers
        destination_path=cfg.paths.reproject.destination_path,  # Output: main path
        config=cfg.paths.reproject.config_file,                 # Output: reprojection json config
    )

    additional_args = get_date_splits(cfg)

    # Convert to dict, and merge
    args = IterableNamespace(**vars(main_args), **vars(additional_args))

    # Initially copy across the source data from `./data/` to the destination
    # `./processed_data/`
    ds, ds_config = init_dataset(args)
    # Reproject and overwrite the copied data
    reproject_datasets_from_config(
        ds_config,
        source_crs=cfg.reproject.source_crs,
        target_crs=cfg.reproject.target_crs,
        shape=cfg.reproject.shape,              # Shape of reprojected grid
        coarsen=1,                              # Coarsen the reprojected grid by this factor
        interpolate_nans=False,                 # Enable nearest neighbour interpolation to fill in missing areas.
        workers=args.workers,
    )
    ds_config.save_config()

    # Create symlink in main run dir
    target = cfg.paths.reproject.destination_path
    run_dir = HydraConfig.get().run.dir
    symlink(target, run_dir)


def preprocess_era5(cfg: DictConfig) -> None:
    """
    Preprocess ERA-5 data using the specified configuration file.

    Args:
        cfg: Hydra preprocess configuration object.
    """
    if Path(cfg.paths.preprocess_era5.config_file).exists():
        logger.warning(
            f"(skipping) preprocess_era5 step already completed previously: {cfg.paths.preprocess_era5.config_file}"
        )
        return

    anom_vars = OmegaConf.to_container(cfg.input.vars.anomaly) if cfg.input.vars.anomaly else None
    abs_vars = OmegaConf.to_container(cfg.input.vars.absolute) if cfg.input.vars.absolute else None

    # Emulating argparse interface (only for interface with preprocess toolbox)
    main_args = IterableNamespace(
        source=cfg.paths.reproject.config_file,                         # Input: reprojected json config
        destination_id=cfg.source_dataset_id,                           # Input: identifier
        split_head=cfg.input.lag_length,                                # Input: Lag
        split_tail=cfg.input.forecast_length,                           # Input: Leadtime
        workers=cfg.workers,                                            # Input: Concurrent workers
        destination_path=cfg.paths.preprocess_era5.destination_path,    # Output: main path
        config=cfg.paths.preprocess_era5.config_file,                   # Output: preprocessed json config
        frequency=cfg.frequency,                                        # Input: Leadtime frequency
        anom=anom_vars,
        abs=abs_vars,
    )

    additional_args = get_date_splits(cfg)

    if cfg.preprocess_type == "train":
        # No `normalisation.{scale,mean}/` path needed when preprocessing for training
        normalisation_path = None
        more_args = {
            "processing_splits": ["train"],
            "ref": normalisation_path,
        }
    else:
        # TODO: For prediction, need to add reference file for normalisation.
        # Reference loader to use same normalisations as the training dataset
        # This should point to the dir that holds `normalisation.scale/` or `normalisation.mean/`
        # e.g.: +train_ref=preprocessed_data/preprocessed/02_normalised_small_test/era5/
        if getattr(cfg, "train_ref", None):
            normalisation_path = cfg.train_ref
        else:
            # If using experiment config file, we should be able to ascertain where
            # the training normalisation file is located
            normalisation_path = (
                Path(cfg.paths.train) / NORMALISATION_SYMLINK_DIR
            )
            if not normalisation_path.exists():
                logging.error("The training normalisation path does"
                              " not exist, have you run training?")
                raise NotADirectoryError(normalisation_path)

        logging.info(f"Training normalisation path: {normalisation_path}")

        more_args = {
            "processing_splits": None,
            "ref": normalisation_path,
        }

    # Convert to dict, and merge
    args = IterableNamespace(**vars(main_args), **vars(additional_args), **more_args)

    ds_config = get_dataset_config_implementation(args.source)
    splits = process_split_args(args, frequency=ds_config.frequency)

    implementation = cfg.preprocess_era5.implementation

    implementation = (
        get_implementation(implementation) if implementation else ERA5PreProcessor
    )

    proc = implementation(
        ds_config,
        args.anom,
        splits,
        args.abs,
        anom_clim_splits=args.processing_splits,
        base_path=args.destination_path,
        config_path=args.config,
        identifier=args.destination_id,
        lag_time=args.split_head,
        lead_time=args.split_tail,
        normalisation_splits=args.processing_splits,
        parallel_opens=False,
        ref_procdir=args.ref,
        smooth_sigma=cfg.preprocess_era5.smooth_sigma,
    ) # pyright: ignore[reportOptionalCall]
    proc.process(config_path=args.config)

    # Create symlink in main run dir
    target = cfg.paths.preprocess_era5.destination_path
    run_dir = HydraConfig.get().run.dir
    symlink(target, run_dir)

    if normalisation_path:
        # Symlink path to normalisation scale/mean for postprocessing use
        abs_normalisation_path = os.path.abspath(normalisation_path)
        src = abs_normalisation_path
        dst = os.path.join(target, "ref_training_dataset")
        logging.info(
            f"Creating symlink to normalisation scale/mean for postprocessing from {src} -> {dst}"
        )
        os.symlink(src, dst)


def preprocess_loader_init(cfg):
    """
    Initialise the data loader configuration file.

    Args:
        cfg: Hydra preprocess configuration object.
    """
    job_name = HydraConfig.get().job.name   # Hydra job name
    data = dict(
        identifier=job_name,
        filenames=dict(),
        sources=dict(),
        masks=dict(),
        channels=dict(),
    )
    output_loader_file = cfg.paths.preprocess.loader_file
    destination_directory = os.path.dirname(output_loader_file)
    if destination_directory:
        os.makedirs(destination_directory, exist_ok=True)

    if not os.path.exists(output_loader_file):
        with open(output_loader_file, "w") as fh:
            fh.write(orjson.dumps(data, option=orjson.OPT_INDENT_2).decode())
        logger.info("Created a configuration {} to build on".format(output_loader_file))
    else:
        logger.error("A loader configuration file already exists, "
                    "delete the file first and run again: "
                    f"{output_loader_file}"
                    )
        raise FileExistsError


def preprocess_loader_add_era5(cfg):
    """
    Add ERA5 data to the preprocessor configuration.

    Args:
        cfg: Hydra preprocess configuration object.
    """
    loader_file = cfg.paths.preprocess.loader_file
    processed_era5_path = cfg.paths.preprocess_era5.config_file
    # `IceNetBaseDataLoader` takes the first path for ground truth
    configurations = [processed_era5_path] # Can add more here later if needed
                                           # like OSI-SAF, CMIP6, etc.

    cfgs = dict()
    filenames = dict()

    for file in configurations:
        with open(file, mode="r") as fh:
            logging.info(f"Configuration {file} being loaded")
            cfg_data = orjson.loads(fh.read())

            if "data" not in cfg_data:
                raise KeyError(
                    f"There's no data element in {file}, that's not right!"
                )
            name = ".".join(fh.name.split(".")[1:-1])
            cfgs[name] = cfg_data["data"]
            filenames[name] = fh.name

    update_config(loader_file, "filenames", filenames)
    update_config(loader_file, "sources", cfgs)


def preprocess_loader_add_mask(cfg):
    """
    Add mask data to the preprocessor configuration.

    Args:
        cfg: Hydra preprocess configuration object.
    """
    loader_file = cfg.paths.preprocess.loader_file
    implementation = cfg.preprocess_mask.implementation
    ground_truth_dataset = cfg.paths.reproject.config_file

    proc_impl = get_implementation(implementation) if implementation else Masks
    ds_config = get_dataset_config_implementation(ground_truth_dataset)

    channel_name = cfg.preprocess_mask.channel_name
    impl_args = (
        ds_config,
        [
            channel_name,
        ],
        channel_name,
    )
    impl_kwargs = {
        "base_path": cfg.paths.mask.destination_path,
        "mask_dataset_config_path": cfg.paths.mask.mask_dataset_config_path,
        "mask_config_path": cfg.paths.mask.mask_config_path,
    }

    processor = proc_impl(*impl_args, **impl_kwargs) # pyright: ignore[reportOptionalCall]
    processor.process()
    update_config(loader_file,
                  "masks",
                  {channel_name: processor.get_config()})


def preprocess_cache(cfg):
    """
    Generate ML dataset cache & config (if train) or just config file (if predict).

    Args:
        cfg: Hydra preprocess configuration object.
    """

    loader_file = cfg.paths.preprocess.loader_file

    dl = CanariMLDataLoaderFactory().create_data_loader(
        cfg.preprocess_cache.implementation,
        loader_file,
        cfg.params.config_name,
        base_path=cfg.paths.cache.destination_path,
        config_path=cfg.paths.cache.config_path,
        dry=None,
        lag_time=cfg.input.lag_length,
        lead_time=cfg.input.forecast_length,
        output_batch_size=cfg.preprocess_cache.output_batch_size,
        pickup=None,    # Does not currently work with continuing generating
        generate_workers=cfg.workers,
        plot=False,
        )

    if cfg.preprocess_type == "predict":
        dl.write_dataset_config_only() # type: ignore
    elif cfg.preprocess_type == "train":
        dl.generate() # type: ignore
    else:
        logger.error(f"Unknown preprocess type: {cfg.preprocess_type}")
        raise ValueError("Unknown preprocess type")

    # Create symlink in main run dir
    target = cfg.paths.cache.destination_path
    run_dir = HydraConfig.get().run.dir
    symlink(target, run_dir)


OmegaConf.register_new_resolver("getcwd", lambda: os.getcwd())
OmegaConf.register_new_resolver("opt_underscore", lambda x: f"_{x}" if x else "")
OmegaConf.register_new_resolver("compute_step_hash", compute_step_hash)
OmegaConf.register_new_resolver("compute_loader_hash", compute_loader_hash)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../../../conf"),
    config_name="preprocess",
)
def preprocess_run(cfg: DictConfig) -> None:
    """
    Run preprocessing steps for train/predict from HYDRA config.

    This function loads a Hydra configuration, and preprocesses the data.

    Args:
        cfg: Hydra auto-loaded configuration.
    """
    cfg_yaml = OmegaConf.to_yaml(cfg)

    logger.info("Loaded HYDRA Configuration YAML")
    logger.info(f"\n{cfg_yaml}")

    preprocess_type = cfg.preprocess_type
    logger.info(f"preprocess_type: {preprocess_type}")

    # Run preprocessing steps
    reproject(cfg)
    preprocess_era5(cfg)

    # Set up a loader config to encapsulate above steps
    print("Init")
    preprocess_loader_init(cfg)
    print("Add era5")
    preprocess_loader_add_era5(cfg)
    print("Add mask")
    preprocess_loader_add_mask(cfg)

    print("Gen cache")
    # Generate dataset cache & config (if train), or, just config file (if predict)
    preprocess_cache(cfg)

    logger.info("All preprocessing steps completed")


def main(preprocess_type: str = "train"):
    """
    Entry point to configure and run the preprocessing pipeline.

    Args:
        preprocess_type: Type of preprocessing, either "train" or "predict".
    """

    logger.info(f"Preprocess type: {preprocess_type}")
    OmegaConf.register_new_resolver("set_preprocess_type", lambda x: preprocess_type)

    # TODO: Code smell, but, hack. Avoid modifying `sys.argv` in future if I can.
    sys.argv.insert(1, f"preprocess_type={preprocess_type}")

    preprocess_run()


if __name__ == "__main__":
    main()
