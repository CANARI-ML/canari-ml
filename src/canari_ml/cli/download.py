import logging
import sys
from datetime import datetime as dt
from pathlib import Path

import hydra
from dateutil.relativedelta import relativedelta
from omegaconf import OmegaConf

from .utils import run_command


logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../../../conf"),
    config_name="config",
)
def download(cfg):
    cfg_yaml = OmegaConf.to_yaml(cfg)

    logger.info("HYDRA Configuration YAML:\n")
    logger.info(cfg_yaml)

    start_date = dt.strptime(cfg.download.date_range.start, "%Y-%m-%d").date()
    end_date = dt.strptime(cfg.download.date_range.end, "%Y-%m-%d").date()
    variables = ",".join(cfg.download.variables)
    levels = ",".join(cfg.download.levels)

    base_command = [
        "download_aws",
        "--config-path",
        cfg.download.output_config_path,
        "-f",
        cfg.download.frequency,
        "-o",
        cfg.download.output_group_by,
        "--workers",
        str(cfg.download_params.workers),
        cfg.download.hemisphere,
    ]

    if cfg.download_params.overwrite_config:
        base_command.append("--overwrite-config")

    if cfg.download_params.delete_cache:
        base_command.append("--delete-cache")

    if cfg.download_params.cache_only:
        base_command.append("--cache-only")

    current_start_date = start_date

    while current_start_date <= end_date:
        command = base_command.copy()
        current_end_date = current_start_date + relativedelta(day=31)

        # Make sure not to go outside of bounds
        if current_end_date > end_date:
            current_end_date = end_date

        month_start_date = current_start_date.strftime("%Y-%m-%d")
        month_end_date = current_end_date.strftime("%Y-%m-%d")

        command.extend(
            [month_start_date, month_end_date, f"'{variables}'", f"'{levels}'"]
        )

        current_start_date += relativedelta(months=1)

        run_command(command)

    logger.info("All downloads completed.")


def main():
    # Set job name for downloader
    sys.argv.append("hydra.job.name=download")

    # Use config to set logger to act like print
    # I've defined it in `conf/hydra/job_logging/basic_message.yaml`
    sys.argv.append("hydra/job_logging=basic_message")

    # Use custom run dir for downloads instead of HYDRA default
    sys.argv.append(
        "hydra.run.dir='outputs/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}'"
    )
    # sys.argv.append("outputs/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}")

    # Disable HYDRA logging for downloader script
    # sys.argv.append("hydra/hydra_logging=none")

    # Disable HYDRA output directory creation
    # sys.argv.append("hydra.output_subdir=null")

    # sys.argv.append("hydra.run.dir='outputs/download'")
    # sys.argv.append("hydra.output_subdir='logs'")
    download()


if __name__ == "__main__":
    main()
