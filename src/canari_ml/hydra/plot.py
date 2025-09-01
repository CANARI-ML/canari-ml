import logging
from pathlib import Path

from omegaconf import DictConfig

import hydra
from hydra.core.hydra_config import HydraConfig

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../conf"),
    config_name="plot",
)
def main(cfg: DictConfig):
    """
    Main function for plotting ERA5 forecast data.

    This function serves as the entry point for generating plots based on
    hydra configuration. It dynamically selects which plotting function to
    execute based on runtime choices specified in the Hydra configuration.

    Args:
        cfg: Hydra configuration parameters containing all necessary
            settings for plotting. This includes paths, plot type selection,
            and any additional parameters required for specific plots.
    """
    from canari_ml.hydra.utils import print_omega_config
    from canari_ml.plotting.forecast import plot_ua700_error
    print_omega_config(cfg)

    # Dynamically selecting postprocessing override as provided by `e.g.: +plot=ua700`
    selected_postprocess = HydraConfig.get().runtime.choices.get("plot")

    logger.info(f"Selected plotting type: {selected_postprocess}")

    # Name of the file selected under `conf/plot/*.yaml`, excluding `.yaml`
    if selected_postprocess == "ua700":
        plot_ua700_error(cfg)
