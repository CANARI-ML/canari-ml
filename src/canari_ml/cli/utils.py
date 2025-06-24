import subprocess

def run_command(command):
    command = [str(v) for v in command]
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command)
