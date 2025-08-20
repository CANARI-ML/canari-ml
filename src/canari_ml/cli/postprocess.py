import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from canari_ml.cli.utils import print_omega_config
from canari_ml.postprocess.predict import create_cf_output

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../../../conf"),
    config_name="postprocess",
)
def main(cfg: DictConfig):
    print_omega_config(cfg)

    # Checking what override was provided by `e.g.: +postprocess=netcdf`
    selected_postprocess = HydraConfig.get().runtime.choices.get("postprocess")

    # Name of the file selected under `conf/postprocess/*.yaml`, excluding `.yaml`
    if selected_postprocess == "netcdf":
        create_cf_output(cfg)
