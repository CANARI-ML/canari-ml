import subprocess
from datetime import datetime as dt
from pathlib import Path

import hydra
from dateutil.relativedelta import relativedelta
from omegaconf import OmegaConf


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../../conf"),
    config_name="config"
)
def main(cfg):
    # print(OmegaConf.to_yaml(cfg))

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

        # Make sure not to outside of bounds
        if current_end_date > end_date:
            current_end_date = end_date

        month_start_date = current_start_date.strftime("%Y-%m-%d")
        month_end_date = current_end_date.strftime("%Y-%m-%d")

        command.extend(
            [month_start_date, month_end_date, f"'{variables}'", f"'{levels}'"]
        )

        current_start_date += relativedelta(months=1)

        print(f"Running command: {' '.join(command)}")
        subprocess.run(command)

    print("All downloads completed.")


if __name__ == "__main__":
    main()
