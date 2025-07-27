import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from canari_ml.models.networks.pytorch import HYDRAPytorchNetwork

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../../../conf"),
    config_name="predict",
)
def predict_run(cfg: DictConfig) -> None:
    """
    Run training based on the provided HYDRA configuration.

    This function loads a Hydra configuration, and trains the model.

    Args:
        cfg: Hydra auto-loaded configuration.
    """

    cfg_yaml = OmegaConf.to_yaml(cfg)

    logger.info("Loaded HYDRA Configuration YAML")
    logger.info(f"\n{cfg_yaml}")

    network = HYDRAPytorchNetwork(cfg)
    network.predict()


def main():
    predict_run() # type: ignore


if __name__ == "__main__":
    main()
