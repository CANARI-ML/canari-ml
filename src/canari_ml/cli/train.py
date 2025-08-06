import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from canari_ml.cli.utils import print_omega_config
from canari_ml.models.networks.pytorch import HYDRAPytorchNetwork

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../../../conf"),
    config_name="train",
)
def train_run(cfg: DictConfig) -> None:
    """
    Run training based on the provided HYDRA configuration.

    This function loads a Hydra configuration, and trains the model.

    Args:
        cfg: Hydra auto-loaded configuration.
    """

    print_omega_config(cfg)

    network = HYDRAPytorchNetwork(cfg, run_type="train")
    network.train()


def main():
    train_run() # type: ignore


if __name__ == "__main__":
    main()
