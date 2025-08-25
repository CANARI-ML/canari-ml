import logging
from datetime import datetime as dt
from pathlib import Path

import hydra
from download_toolbox.data.aws import AWSDatasetConfig, AWSDownloader
from download_toolbox.location import Location
from download_toolbox.time import Frequency
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../../../conf"),
    config_name="download",
)
def download(cfg: DictConfig):
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

    hemisphere = cfg.hemisphere

    location = Location(
        name=hemisphere,
        north=hemisphere == "north",
        south=hemisphere == "south",
    )

    dataset = AWSDatasetConfig(
        levels=var_levels,
        location=location,
        var_names=var_names,
        frequency=getattr(Frequency, cfg.frequency),
        output_group_by=getattr(Frequency, cfg.output_group_by),
        config_path=cfg.paths.download.config_file, # Output json config path to use
        overwrite=cfg.overwrite_config,
    )

    # If given just a single date, convert to list
    start_dates = [cfg.dates.start] if isinstance(cfg.dates.start, str) else cfg.dates.start
    end_dates = [cfg.dates.end] if isinstance(cfg.dates.end, str) else cfg.dates.end
    
    # Make sure the length of the start and end dates are the same
    if len(start_dates)!= len(end_dates):
        raise ValueError("Start and end dates must be the same length")

    logger.debug(f"Dates type: {type(cfg.dates.start)}")
    logger.debug(f"Dates: {cfg.dates.start}")

    for start_date, end_date in zip(start_dates, end_dates):
        logger.info("Downloading between {} and {}".format(start_date, end_date))
        start_date = dt.strptime(start_date, "%Y-%m-%d").date()
        end_date = dt.strptime(end_date, "%Y-%m-%d").date()
        aws = AWSDownloader(
            dataset,
            start_date=start_date,
            end_date=end_date,
            delete_cache=cfg.delete_cache,
            cache_only=cfg.cache_only,
            compress=None,
            max_threads=cfg.workers,
            request_frequency=getattr(Frequency, cfg.output_group_by),
        )
        aws.download()

        dataset.save_data_for_config(
            source_files=aws.files_downloaded,
            var_filter_list=["lambert_azimuthal_equal_area"],
        )


if __name__ == "__main__":
    download()
