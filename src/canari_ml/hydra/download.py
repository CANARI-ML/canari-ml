import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from canari_ml.download.era5 import download_daily

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../conf"),
    config_name="download",
)
def main(cfg: DictConfig) -> None:
    """Download ERA5 reanalysis data using AWS downloader in `download-toolbox`.

    Processes configuration settings, sets up the download parameters
    and downloads daily ERA5 data from AWS mirror.

    Args:
        cfg: Hydra configuration parameters.
    """
    cfg_yaml = OmegaConf.to_yaml(cfg)

    logger.info("Loaded HYDRA Configuration YAML")
    logger.info(f"\n{cfg_yaml}")

    logger.info("AWS Data Downloading")

    vars = [cfg.vars] if isinstance(cfg.vars, str) else cfg.vars
    levels = [cfg.levels] if isinstance(cfg.levels, int) else cfg.levels

    var_names = []
    var_levels = []
    for var_name, var_level in zip(vars, levels):
        var_names.append(var_name)
        if not var_level:
            var_levels.append(None)
        elif isinstance(var_level, int):
            var_levels.append([var_level])
        else:
            var_levels.append([int(level) for level in var_level.split("|")])

    download_daily(
        var_names=var_names,
        var_levels=var_levels,
        start_dates=cfg.dates.start,
        end_dates=cfg.dates.end,
        hemisphere=cfg.hemisphere,
        frequency=cfg.frequency,
        output_group_by=cfg.output_group_by,
        config_path=cfg.paths.download.config_file,
        overwrite=cfg.overwrite_config,
        delete_cache=cfg.delete_cache,
        cache_only=cfg.cache_only,
        compress=0,
        workers=cfg.workers,
    )


if __name__ == "__main__":
    main()
