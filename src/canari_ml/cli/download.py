import sys
from datetime import datetime as dt
from pathlib import Path

import hydra
from dateutil.relativedelta import relativedelta
from omegaconf import OmegaConf

from .utils import run_command


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../../../conf"),
    config_name="config",
)
def download(cfg):
    Path("logs").mkdir(exist_ok=True)
    now = dt.now()
    log_file = f"logs/download.{now:%Y-%m-%d_%H-%M-%S}.log"

    cfg_yaml = OmegaConf.to_yaml(cfg)

    print("HYDRA Configuration YAML:\n")
    print(cfg_yaml)

    with open(log_file, "w") as f:
        f.write("HYDRA Configuration YAML:\n\n")
        f.write(cfg_yaml)
        f.write("\n\n")
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

            run_command(command, log_file=f)

        end_msg = "All downloads completed."
        f.write(end_msg)
        f.write("\n\n")

        print(end_msg)


def main():
    # Disable HYDRA logging for downloader script
    # sys.argv.append("hydra/hydra_logging=none")

    # Disable HYDRA output directory creation
    # sys.argv.append("hydra.output_subdir=null")

    # sys.argv.append("hydra.run.dir='outputs/download'")
    # sys.argv.append("hydra.output_subdir='logs'")
    download()


if __name__ == "__main__":
    main()
