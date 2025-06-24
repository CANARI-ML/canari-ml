import subprocess


def run_command(command, log_file=None):
    command = [str(v) for v in command]
    cmd_str = f"Running command: {' '.join(command)}\n{'_' * 75}\n\n"
    print(cmd_str)
    if log_file:
        log_file.write(cmd_str)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:  # type: ignore
        print(line, end="")
        if log_file:
            log_file.write(line)
