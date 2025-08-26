import logging
from datetime import datetime as dt

from download_toolbox.data.aws import AWSDatasetConfig, AWSDownloader
from download_toolbox.location import Location
from download_toolbox.time import Frequency

logger = logging.getLogger(__name__)


def download_daily(
    var_names: list[str],
    var_levels: list[int],
    start_dates: list[str] | str,
    end_dates: list[str] | str,
    hemisphere: str,
    frequency: str,
    output_group_by: str,
    config_path: str,
    overwrite: bool,
    delete_cache: bool,
    cache_only: bool,
    compress: int = 0,
    workers: int = 1,
) -> None:
    """Download ERA5 daily reanalysis dataset from AWS S3 using download-toolbox.

    Processes configuration settings and downloads daily ERA5
    data for the specified variables, pressure levels, and date range.

    Args:
        var_names: List of ERA5 variables to download
        var_levels: Corresponding list of pressure levels for the
            variables
        start_dates: Start dates in "YYYY-MM-DD" format
        end_dates: End dates in "YYYY-MM-DD" format, matching length
            with start_dates
        hemisphere: Either "north" or "south"
        frequency: Temporal resolution of data (e.g., "daily")
        output_group_by: Grouping frequency for output files
        config_path: Path to save configuration file
        overwrite: Whether to overwrite existing files
        delete_cache: Delete temporary cache files after download
        cache_only: Only use cached files, no download
        compress (optional): Compression level (0-9)
            Defaults to 0.
        workers (optional): Number of download workers.
            Defaults to 1.
    """

    location = Location(
        name=hemisphere,
        north=hemisphere == "north",
        south=hemisphere == "south",
    )

    dataset = AWSDatasetConfig(
        levels=var_levels,
        location=location,
        var_names=var_names,
        frequency=getattr(Frequency, frequency),
        output_group_by=getattr(Frequency, output_group_by),
        config_path=config_path,  # Output json config path to use
        overwrite=overwrite,
    )

    # If given just a single date, convert to list
    start_dates = [start_dates] if isinstance(start_dates, str) else start_dates
    end_dates = [end_dates] if isinstance(end_dates, str) else end_dates

    # Make sure the length of the start and end dates are the same
    if len(start_dates) != len(end_dates):
        raise ValueError("Start and end dates must be the same length")

    logger.debug(f"Dates type: {type(start_dates)}")
    logger.debug(f"Dates: {start_dates}")

    for start_date, end_date in zip(start_dates, end_dates):
        logger.info("Downloading between {} and {}".format(start_date, end_date))
        start_date = dt.strptime(start_date, "%Y-%m-%d").date()
        end_date = dt.strptime(end_date, "%Y-%m-%d").date()
        aws = AWSDownloader(
            dataset,
            start_date=start_date,
            end_date=end_date,
            delete_cache=delete_cache,
            cache_only=cache_only,
            compress=compress,
            max_threads=workers,
            request_frequency=getattr(Frequency, output_group_by),
        )
        aws.download()

        dataset.save_data_for_config(
            source_files=aws.files_downloaded,
            var_filter_list=["lambert_azimuthal_equal_area"],
        )
