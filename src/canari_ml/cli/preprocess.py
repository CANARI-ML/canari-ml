import logging
import sys
from hashlib import shake_256
from pathlib import Path

import hydra
import orjson
from omegaconf import DictConfig, ListConfig, OmegaConf

from .utils import run_command

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("subtract", lambda x, y: x - y)


def get_cmd_props(step_cfg: DictConfig) -> tuple:
    """Get command properties and generate a hash for the step.

    Extracts necessary information from the given configuration dictionary,
    removes output paths from optional variables, constructs an input
    dictionary, and generates a SHAKE256 hash of fixed length to uniquely identify
    this step.

    Args:
        cfg: The configuration dictionary containing command, positional, and
        optional arguments.

    Returns:
        A tuple containing the command, positional arguments, optional arguments,
        and the generated hash for the step.
    """
    """Ref: https://www.doc.ic.ac.uk/~nuric/posts/coding/how-to-hash-a-dictionary-in-python/"""
    cmd: str = step_cfg.get("command")
    positional: ListConfig = step_cfg.get("positional", ListConfig(content=[]))
    optional: DictConfig = step_cfg.get("optional", DictConfig(content={}))

    # Mask: Remove output paths from optional variables
    optional_dict: dict = OmegaConf.to_container(optional, resolve=False)
    optional_resolved_dict: dict = OmegaConf.to_container(optional, resolve=True)
    optional_masked = {}
    for k, v in optional_dict.items():
        # Check for and remove keys with values of something like
        # ` ${preprocess_reproject.output.config_path}`
        if isinstance(v, str) and ".output." in v:
                logging.debug(f"Removing output dirs in hash generation: {k}: {v}")
                continue
        optional_masked[k] = optional_resolved_dict[k]

    step_inputs = {
        "cmd": cmd,
        "positional": OmegaConf.to_container(positional, resolve=True),
        "optional": optional_masked,
    }
    step_inputs_dict = step_inputs
    cmd_string = orjson.dumps(step_inputs_dict, option=orjson.OPT_SORT_KEYS)
    # Using shake_256 instead of shake_128 for hopefully higher collision resistance
    # Especially considering I'm limiting hash length
    step_hash = shake_256(cmd_string).hexdigest(length=4) # Will generate a 2xlength hash
    return cmd, positional, optional, step_hash


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../../../conf"),
    config_name="config",
)
def preprocess_run_commands(cfg: DictConfig) -> None:
    """
    Run preprocessing commands based on the provided HYDRA configuration.

    This function loads a Hydra configuration, selects the appropriate steps to run
    (either for training or prediction), and then executes each step's command with
    its corresponding arguments.

    Args:
        cfg: Hydra auto-loaded configuration.
    """
    cfg_yaml = OmegaConf.to_yaml(cfg)

    logger.info("Loaded HYDRA Configuration YAML:\n")
    logger.info(cfg_yaml)

    # Select high level dict of preprocess steps to run
    # Either for training or prediction
    steps = cfg.preprocess_train_steps  # or cfg.preprocess_predict_steps

    logger.info("\nRunning preprocessing steps:")
    for step_key, step_cfg in steps.items():
        step_name = step_cfg.get("name", "Unnamed step")

        # Get command and arguments, and hash them
        cmd, positional, optional, step_hash = get_cmd_props(step_cfg)

        logger.info(f"\nRunning step: {step_name}")
        logger.info(f"Command: {cmd}")
        logger.info(f"Positional args: {positional}")
        logger.info(f"Optional args: {optional}")
        logger.info(f"Step hash: {step_hash}\n")

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
