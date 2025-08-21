import sys
import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from canari_ml.cli.utils import print_omega_config
from canari_ml.postprocess.predict import create_cf_output

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../../../conf"),
    config_name="postprocess",
)
def postprocess_main(cfg: DictConfig):
    print_omega_config(cfg)

    # Dynamically selecting postprocessing override as provided by `e.g.: +postprocess=netcdf`
    selected_postprocess = HydraConfig.get().runtime.choices.get("postprocess")

    logger.info(f"Selected postprocessing type: {selected_postprocess}")

    # Name of the file selected under `conf/postprocess/*.yaml`, excluding `.yaml`
    if selected_postprocess == "netcdf":
        create_cf_output(cfg)


def main():
    # # TODO: Code smell, but, hack. Avoid modifying `sys.argv` in future if I can.
    sys.argv.append("++preprocess_type=predict")
    OmegaConf.register_new_resolver("set_preprocess_type", lambda x: "predict")

    postprocess_main()
