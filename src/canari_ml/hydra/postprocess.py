import sys
import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from canari_ml.hydra.utils import print_omega_config
from canari_ml.postprocess.predict import create_cf_output

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../conf"),
    config_name="postprocess",
)
def postprocess_main(cfg: DictConfig):
    """
    Main function for ERA5 forecast data postprocessing.

    Processes Hydra configuration settings, selects a postprocessing method
    based on runtime choices, and executes the corresponding postprocessing
    task.

    Args:
        cfg: Hydra configuration parameters for postprocessing.
    """
    print_omega_config(cfg)

    # Dynamically selecting postprocessing override as provided by `e.g.: +postprocess=netcdf`
    selected_postprocess = HydraConfig.get().runtime.choices.get("postprocess")

    logger.info(f"Selected postprocessing type: {selected_postprocess}")

    # Name of the file selected under `conf/postprocess/*.yaml`, excluding `.yaml`
    if selected_postprocess == "netcdf":
        create_cf_output(cfg)


def main():
    """Entry point for initialising ERA5 forecast postprocessing.

    This function sets up initial configuration overrides and calls the
    `postprocess_main` function to execute the postprocessing task.
    """
    OmegaConf.register_new_resolver("set_preprocess_type", lambda x: "predict")

    # # TODO: Code smell, but, hack. Avoid modifying `sys.argv` in future if I can.
    sys.argv.insert(1, "++preprocess_type=predict")

    postprocess_main()
