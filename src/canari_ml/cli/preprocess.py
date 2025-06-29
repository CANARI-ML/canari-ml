import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .utils import run_command

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("subtract", lambda x, y: x - y)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../../../conf"),
    config_name="config",
)
def preprocess_run_commands(cfg: DictConfig):
    cfg_yaml = OmegaConf.to_yaml(cfg)

    logger.info("Loaded HYDRA Configuration YAML:\n")
    logger.info(cfg_yaml)

    # Select high level dict of preprocess steps to run
    # Either for training or prediction
    steps = cfg.preprocess_train_steps  # or cfg.preprocess_predict_steps

    logger.info("\nRunning preprocessing steps:")
    for step_key, step in steps.items():
        step_name = step.get("name", "Unnamed step")
        cmd = step.get("command")
        positional = step.get("positional", [])
        optional = step.get("optional", {})

        logger.info(f"\nRunning step: {step_name}")
        logger.info(f"Command: {cmd}")
        logger.info(f"Positional args: {positional}")
        logger.info(f"Optional args: {optional}\n")

        command = [cmd] + [str(arg) for arg in positional]
        for opt_key, opt_val in optional.items():
            if opt_val != "":
                command.append(opt_key)
                if opt_val is not True:
                    command.append(str(opt_val))

        if cfg.preprocess_params.verbose:
            command.append("-v")

        run_command(command)

        logger.info("All preprocessing completed.")


def main():
    # Set job name for downloader
    sys.argv.append("hydra.job.name=preprocess")

    # Use config to set logger to act like print
    # I've defined it in `conf/hydra/job_logging/basic_message.yaml`
    sys.argv.append("hydra/job_logging=basic_message")

    # Use custom run dir for downloads instead of HYDRA default
    sys.argv.append(
        "hydra.run.dir='outputs/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}'"
    )
    # sys.argv.append("outputs/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}")
    preprocess_run_commands()


if __name__ == "__main__":
    main()
