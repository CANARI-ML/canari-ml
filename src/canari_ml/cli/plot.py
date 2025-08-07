import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from canari_ml.cli.utils import print_omega_config
from canari_ml.plotting.forecast import plot_ua700_error

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../../../conf"),
    config_name="plot",
)
def main(cfg: DictConfig):
    print_omega_config(cfg)

    # Checking what override was provided by `e.g.: +postprocess=plot_ua700`
    selected_postprocess = HydraConfig.get().runtime.choices.get("plot")

    # Name of the option defined within a conf/postprocess/*.yaml file
    if selected_postprocess == "ua700":
        plot_ua700_error(cfg)

