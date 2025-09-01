import logging
import os
import subprocess

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s\t-  %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Main options
main_command = "canari_ml"
output_dir = "docs/help"

# Help commands to capture
commands = [
    (f"{main_command}", "general.md"),
    (f"{main_command} download", "download.md"),
    (f"{main_command} preprocess train", "preprocess.md"),
    (f"{main_command} train", "train.md"),
    (f"{main_command} predict", "predict.md"),
    (f"{main_command} postprocess", "postprocess.md"),
    (f"{main_command} plot", "plot.md"),
]

os.makedirs(output_dir, exist_ok=True)


def run_command(command):
    try:
        result = subprocess.run(
            command, shell=True, check=True, text=True, capture_output=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.info(f"Error running command '{command}': {e}")
        return ""


# Generate markdown for each command's --help output
for command, filename in commands:
    logger.info(f"Generating documentation for: `{command}`")
    help_output = run_command(f"{command} --help")
    if help_output:
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(f"# `{command}` CLI Help\n\n")
            f.write(
                "Run the following command to get the help information for "
                f"`{command}` command:\n"
            )
            f.write("``` console\n")
            f.write(f"$ {command} --help\n\n")
            f.write(help_output)
            f.write("```\n")

logger.info("All help documentation generated successfully!")
