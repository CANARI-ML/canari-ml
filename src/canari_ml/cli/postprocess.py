import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from canari_ml.cli.utils import print_omega_config
from canari_ml.postprocess.predict import create_cf_output

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../../../conf"),
    config_name="postprocess",
)
def out_netcdf(cfg: DictConfig):
    print_omega_config(cfg)

    create_cf_output(cfg)

