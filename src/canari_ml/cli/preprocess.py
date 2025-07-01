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


OmegaConf.register_new_resolver("subtract", lambda x, y: x - y)
OmegaConf.register_new_resolver("compute_step_hash", compute_step_hash)
OmegaConf.register_new_resolver("compute_loader_hash", compute_loader_hash)


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
        cfg.preprocess_main.params.config_file,  # Loader JSON config file
        cfg.preprocess_cache.output.data_dir,  # Dataset Cache directory
    ]
    run_dir = cfg.preprocess_main.hydra_properties.run_dir

    for target in main_output_nodes:
        symlink_path = os.path.join(run_dir, os.path.basename(target))
        if not os.path.exists(target):
            raise ValueError(
                f"Target path `{target}` does not exist.\nError in preprocessing?"
            )
        elif os.path.exists(symlink_path):
            logger.warning(f"Symlink already exists: `{symlink_path}`, skipping.")
        else:
            logger.info(f"Symlinking:\n\t`{target}` -> `{symlink_path}`")
            os.symlink(target, symlink_path)

    logger.info("Done.")


def main():
    # Set job name for downloader
    sys.argv.append("hydra.job.name=${preprocess_main.hydra_properties.job_name}")

    # Use config to set logger to act like print
    # I've defined it in `conf/hydra/job_logging/basic_message.yaml`
    sys.argv.append("hydra/job_logging=basic_message")

    # Use custom run dir for downloads instead of HYDRA default
    sys.argv.append("hydra.run.dir=${preprocess_main.hydra_properties.run_dir}")
    # sys.argv.append("outputs/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}")
    preprocess_run_commands()


if __name__ == "__main__":
    main()
