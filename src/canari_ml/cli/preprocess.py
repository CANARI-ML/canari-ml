import logging
import os
import sys
from hashlib import shake_256
from pathlib import Path

import hydra
import orjson
from omegaconf import DictConfig, ListConfig, OmegaConf

from .utils import run_command

logger = logging.getLogger(__name__)


def generate_hash(inputs: list) -> str:
    """Generate a SHAKE-256 hash from input data.

    Reference [this page](Ref: https://www.doc.ic.ac.uk/~nuric/posts/coding/how-to-hash-a-dictionary-in-python/)

    Args:
        inputs: List of input data to be hashed. The contents are serialized
            using orjson with sorted keys for deterministic output.

    Returns:
        4-character hexadecimal digest of the hash.
    """
    hash_input = orjson.dumps(inputs, option=orjson.OPT_SORT_KEYS)
    hash_value = shake_256(hash_input).hexdigest(length=4)
    return hash_value


def compute_step_hash(nodes: ListConfig) -> str:
    """Compute hash from multiple OmegaConf nodes.

    Args:
        nodes: List of OmegaConf nodes to process.

    Returns:
        Hash generated from the combined input data of all nodes
        converted to dicts.
    """
    combined_inputs = []

    for node in nodes:
        node_dict = OmegaConf.to_container(node, resolve=True)
        combined_inputs.append(node_dict)

    return generate_hash(combined_inputs)


def compute_loader_hash(steps: ListConfig) -> str:
    """Compute hash from step hashes of multiple OmegaConf nodes.

    Args:
        steps: List of OmegaConf nodes containing step_hash attributes.
            Each step's step_hash is collected and used as input for the
            final hash.

    Returns:
        Hash generated from the combined step hashes of all provided steps.
    """
    combined_inputs = []

    for step in steps:
        combined_inputs.append(step.step_hash)

    return generate_hash(combined_inputs)


def run_steps(cfg: DictConfig, steps: DictConfig):
    """
    Executes preprocessing steps as defined in HYDRA configuration.

    This function processes each step in the `steps` dictionary, checking if outputs
    already exist (based on the `skippable` flag) and runs commands only when necessary.
    After processing all steps, it creates symlinks from main output directories
    to the HYDRA run directory for easy access.

    Args:
        cfg: Main configuration object containing global settings,
             including paths and verbosity flags.
        steps: DictConfig of preprocessing steps. Each step contains
               a command, positional arguments, optional arguments,
               and output locations.

    Raises:
        ValueError: If any target path for symlinking does not exist, likely
                    due to potential failure in preprocessing.

    Notes:
        - Skippable steps are determined by the presence of existing outputs.
        - Commands are constructed from `command`, `positional`, and `optional`
          fields in each step configuration.
        - Verbose mode is enabled via `cfg.preprocess_main.params.verbose`.
    """
    cfg_yaml = OmegaConf.to_yaml(cfg)

    logger.info("Loaded HYDRA Configuration YAML:\n")
    logger.info(cfg_yaml)

    logger.info("\nRunning preprocessing steps:")
    for step_key, step_cfg in steps.items():
        step_name = step_cfg.get("name", "Unnamed step")

        # Get command and arguments, and hash them
        # cmd, positional, optional = get_cmd_props(cfg, step_cfg)
        cmd: str = step_cfg.get("command")
        positional: ListConfig = step_cfg.get("positional", ListConfig(content=[]))
        optional: DictConfig = step_cfg.get("optional", DictConfig(content={}))
        optional: DictConfig = step_cfg.get("optional", DictConfig(content={}))
        step_hash: DictConfig = step_cfg.get("step_hash", DictConfig(content={}))

        logger.info(f"\nRunning step: {step_name}")
        logger.info(f"Command: {cmd}")
        logger.info(f"Positional args: {positional}")
        logger.info(f"Optional args: {optional}")
        if step_hash:
            logger.info(f"Step hash: {step_hash}\n")

        outputs_exist = False
        for output_location in step_cfg.parent.output.values():
            # Check if skippable key exists
            # Necessary for masks as an example which writes to initialised loader config
            skippable = step_cfg.get("skippable", True)
            if os.path.exists(output_location) and skippable:
                logger.warning(
                    f"Skipping, output already exists: \n\t{output_location}"
                )
                outputs_exist = True
                break   # No need to check further

        # Skip running the command only if all outputs already exist
        if outputs_exist:
            continue

        command = [cmd] + [str(arg) for arg in positional]
        for opt_key, opt_val in optional.items():
            if opt_val != "":
                command.append(str(opt_key))
                if opt_val is not True:
                    command.append(str(opt_val))

        if cfg.preprocess_main.params.verbose:
            command.append("-v")

        run_command(command)

    logger.info("All preprocessing completed.")

    # Symlink all high-level outputs to HYDRA run dir
    main_output_nodes = [
        cfg.preprocess_reproject.output.data_dir,  # Reprojection data dir
        cfg.preprocess_era5.output.data_dir,  # ERA5 Preprocessed data dir
        cfg.preprocess_main.output.config_file,  # Loader JSON config file
        cfg.preprocess_cache.output.data_dir,  # Dataset Cache directory
    ]
    run_dir = cfg.preprocess_main.hydra_properties.run_dir

    for target in main_output_nodes:
        symlink_path = os.path.join(run_dir, os.path.basename(target))
        if not os.path.exists(target):
            logger.error(
                f"Target path `{target}` does not exist.\nError in preprocessing?"
            )
        elif os.path.exists(symlink_path):
            logger.warning(f"Symlink already exists: `{symlink_path}`, skipping.")
        else:
            logger.info(f"Symlinking:\n\t`{target}` -> `{symlink_path}`")
            os.symlink(target, symlink_path)


    logger.info("\n" + "=="*50 + "\nPreprocessing Completed. Tata!\n" + "=="*50)


OmegaConf.register_new_resolver("subtract", lambda x, y: x - y)
OmegaConf.register_new_resolver("compute_step_hash", compute_step_hash)
OmegaConf.register_new_resolver("compute_loader_hash", compute_loader_hash)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../../../conf"),
    config_name="config",
)
def preprocess_run_train_commands(cfg: DictConfig) -> None:
    """
    Run preprocessing commands based on the provided HYDRA configuration.

    This function loads a Hydra configuration, selects the appropriate steps to run
    (either for training or prediction), and then executes each step's command with
    its corresponding arguments.

    Args:
        cfg: Hydra auto-loaded configuration.
    """
    # Select high level dict of preprocess steps to run
    # Either for training or prediction
    steps = cfg.preprocess_train_steps  # or cfg.preprocess_predict_steps
    run_steps(cfg, steps)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "../../../conf"),
    config_name="config",
)
def preprocess_run_predict_commands(cfg: DictConfig) -> None:
    """
    Run preprocessing commands based on the provided HYDRA configuration.

    This function loads a Hydra configuration, selects the appropriate steps to run
    (either for training or prediction), and then executes each step's command with
    its corresponding arguments.

    Args:
        cfg: Hydra auto-loaded configuration.
    """
    # Select high level dict of preprocess steps to run
    # Either for training or prediction
    steps = cfg.preprocess_predict_steps  # or cfg.preprocess_predict_steps
    run_steps(cfg, steps)


def main(preprocess_type="train"):
    # Set job name for downloader
    sys.argv.append("hydra.job.name=${preprocess_main.hydra_properties.job_name}")

    # Use config to set logger to act like print
    # I've defined it in `conf/hydra/job_logging/basic_message.yaml`
    sys.argv.append("hydra/job_logging=basic_message")

    # Use custom run dir for downloads instead of HYDRA default
    sys.argv.append("hydra.run.dir=${preprocess_main.hydra_properties.run_dir}")
    # sys.argv.append("outputs/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}")
    if preprocess_type == "train":
        preprocess_run_train_commands()
    elif preprocess_type == "predict":
        preprocess_run_predict_commands()
    else:
        raise ValueError(f"{preprocess_type} is not a valid preprocess type")


if __name__ == "__main__":
    main()
