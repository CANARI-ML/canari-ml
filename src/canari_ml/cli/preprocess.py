from pathlib import Path

import hydra
from omegaconf import OmegaConf

from .utils import run_command


def preprocess_init(cfg):
    # Initialise preprocess-toolbox
    command = [
        "preprocess_loader_init",
        cfg.preprocess_params.config_name,
    ]

    if cfg.preprocess_params.verbose:
        command.append("-v")

    run_command(command)


def preprocess_reproject(cfg):
    # Reproject data to LAEA
    command = [
        "canari_ml_preprocess_reproject",
        "--workers",
        cfg.preprocess_params.workers,
        "--processing-splits",
        cfg.preprocess_splits.splits,
        "--split-names",
        cfg.preprocess_splits.split_names,
        "-ss",  # TODO: Using short option due to inconsistency bug, preprocess-toolbox #36
        f"'{cfg.preprocess_splits.split_starts}'",
        "-se",  # TODO: Using short option due to inconsistency bug, preprocess-toolbox #36
        f"'{cfg.preprocess_splits.split_ends}'",
        "--split-head",
        cfg.preprocess_splits.split_head,
        "--split-tail",
        cfg.preprocess_splits.split_tail,
        "--source-crs",
        cfg.preprocess_reproject.source_crs,
        "--target-crs",
        cfg.preprocess_reproject.target_crs,
        "--shape",
        cfg.preprocess_reproject.shape,
        "--config-path",
        cfg.preprocess_reproject.output_config_path,
        cfg.preprocess_reproject.source,
        cfg.preprocess_reproject.destination_id,
    ]

    if cfg.preprocess_params.verbose:
        command.append("-v")

    run_command(command)


def preprocess_era5(cfg):
    # Preprocess/Normalise ERA5 data
    command = [
        "preprocess_dataset",
        "--processing-splits",
        cfg.preprocess_splits.splits,
        "--split-names",
        cfg.preprocess_splits.split_names,
        "-ss",  # TODO: Using short option due to inconsistency bug, preprocess-toolbox #36
        f"'{cfg.preprocess_splits.split_starts}'",
        "-se",  # TODO: Using short option due to inconsistency bug, preprocess-toolbox #36
        f"'{cfg.preprocess_splits.split_ends}'",
        "--split-head",
        cfg.preprocess_splits.split_head,
        "--split-tail",
        cfg.preprocess_splits.split_tail,
        "--config-path",
        cfg.preprocess_era5.output_config_path,
        "--implementation",
        cfg.preprocess_era5.__target__,
        cfg.preprocess_era5.source,
        cfg.preprocess_era5.destination_id,
    ]

    if cfg.preprocess_params.verbose:
        command.append("-v")

    absolute_vars = cfg.inputs.absolute_vars
    anomaly_vars = cfg.inputs.anomaly_vars

    if absolute_vars:
        command.extend(["--abs", absolute_vars])
    if anomaly_vars:
        command.extend(["--anom", anomaly_vars])

    run_command(command)


def preprocess_add_era5(cfg):
    command = [
        "preprocess_add_processed",
        cfg.preprocess_params.config_name,
        cfg.preprocess_era5.output_config_path,
    ]

    if cfg.preprocess_params.verbose:
        command.append("-v")

    run_command(command)


def preprocess_add_hemisphere_mask(cfg):
    command = [
        "preprocess_add_mask",
        cfg.preprocess_params.config_name,
        cfg.preprocess_reproject.output_config_path,
        cfg.preprocess_mask.name,
        cfg.preprocess_mask.__target__,
    ]

    if cfg.preprocess_params.verbose:
        command.append("-v")

    run_command(command)


def preprocess_add_region_weights(cfg):
    command = [
        "canari_ml_preprocess_add_region_weights",
        cfg.preprocess_params.config_name,
        cfg.preprocess_reproject.output_config_path,
        cfg.preprocess_region_weight.name,
        cfg.preprocess_region_weight.__target__,
        "--base-weight",
        cfg.preprocess_region_weight.base_weight,
        "--weight-smoothing-sigma",
        cfg.preprocess_region_weight.weight_smoothing_sigma,
    ]

    for region_weight in cfg.preprocess_region_weight.region_weights:
        command.extend(
            [
                "--region-weights",
                region_weight,
            ]
        )

    if cfg.preprocess_params.verbose:
        command.append("-v")

    run_command(command)


def create_cached_dataset(cfg):
    command = [
        "canari_ml_dataset_create",
        "--output-batch-size",
        cfg.preprocess_cache.output_batch_size,
        "--workers",
        cfg.preprocess_params.workers,
        "--forecast-length",
        cfg.inputs.forecast_length,
        cfg.preprocess_params.config_file,
        cfg.preprocess_cache.dataset_name,
    ]

    if cfg.preprocess_params.verbose:
        command.append("-v")

    if cfg.preprocess_cache.plot:
        command.append("--plot")

    run_command(command)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../../../conf"),
    config_name="config",
)
def main(cfg):
    OmegaConf.register_new_resolver("subtract", lambda x, y: x - y)
    # print(OmegaConf.to_yaml(cfg))

    preprocess_init(cfg)

    preprocess_reproject(cfg)
    preprocess_era5(cfg)

    preprocess_add_era5(cfg)
    preprocess_add_hemisphere_mask(cfg)
    preprocess_add_region_weights(cfg)

    create_cached_dataset(cfg)
    print("All preprocessing completed.")


if __name__ == "__main__":
    main()
