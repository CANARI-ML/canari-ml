import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

import hydra
from canari_ml.preprocess.utils import preprocess_register_resolvers

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../conf"),
    config_name="train",
)
def train_run(cfg: DictConfig) -> None:
    """
    Run training based on the provided HYDRA configuration.

    This function loads a Hydra configuration, and trains the model.

    Args:
        cfg: Hydra auto-loaded configuration.
    """
    from canari_ml.hydra.utils import print_omega_config
    print_omega_config(cfg)

    from canari_ml.models.networks.pytorch import HYDRAPytorchNetwork
    network = HYDRAPytorchNetwork(cfg, run_type="train")
    network.train()


def main():
    OmegaConf.register_new_resolver("set_preprocess_type", lambda: "train")
    preprocess_register_resolvers()
    train_run() # type: ignore


if __name__ == "__main__":
    main()
