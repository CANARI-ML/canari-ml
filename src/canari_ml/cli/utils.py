import importlib
import logging
import subprocess


def run_command(command):
    logger = logging.getLogger(__name__)

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


def dynamic_import(path):
    # Split into module and class name
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
