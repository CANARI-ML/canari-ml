import importlib
import logging
import subprocess

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def get_hydra_config_root_path():
    config_path = importlib.resources.files("canari_ml").joinpath("conf")
    return str(config_path)


def run_command(command: list):
    """
    Run a command and log its output.

    Args:
        command: List of strings representing the command to execute.

    Raises:
        RuntimeError: If the command exits with a non-zero return code.
    """
    command = [str(v) for v in command]
    cmd_str = f"\n\nRunning command: {' '.join(command)}\n{'_' * 75}\n\n"
    logger.info(cmd_str)


    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:  # type: ignore
        logger.info(line.strip())

    process.wait()

    if process.returncode != 0:
        raise RuntimeError("Command failed with exit code %d" % process.returncode)


def dynamic_import(path: str):
    """
    Dynamically import a class or function from a module.

    Takes a fully qualified name (e.g., 'module.class') and imports the
    specified class or function.

    Args:
        path: String containing the fully qualified name of the object
            to import, in the format 'module.class'.

    Returns:
        The imported class or function.
    """
    # Split into module and class name
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def print_omega_config(cfg: DictConfig):
    """Print a HYDRA configuration as YAML.

    This function converts the given OmegaConf DictionaryConfig to
    YAML format and logs it to stdout.

    Args:
        cfg: The Hydra configuration to print.
    """
    cfg_yaml = OmegaConf.to_yaml(cfg)

    logger.info("Loaded HYDRA Configuration YAML")
    logger.info(f"\n{cfg_yaml}")
