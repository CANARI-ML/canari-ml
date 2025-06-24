import logging
import subprocess

def run_command(command):
    logger = logging.getLogger(__name__)

    command = [str(v) for v in command]
    cmd_str = f"Running command: {' '.join(command)}\n{'_' * 75}\n\n"
    logger.info(cmd_str)

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:  # type: ignore
        logger.info(line.strip())
