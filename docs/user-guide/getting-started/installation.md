# Installation

# Setting Up Your Development Environment

To get started with using `canari-ml`, you will need to have Python set-up on your system. This guide walks you through creating either a Mamba/Conda environment or a virtual environment using `venv`, and installing the latest development versions of the codebase directly from Git. In future, this will be directly installable via pip.

## Creating a Virtual Python Environment

### Using Mamba/Conda

1. **Install Miniforge3** (if not already installed):

    Follow instructions for your system from the official [Miniforge3 GitHub page](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install).

2. **Create and activate a new environment**:

    You can replace the `mamba` command with `conda` if you prefer.

    ``` console
    mamba create -n canari_ml python=3.11 -y
    mamba activate canari_ml
    ```

### Using Python's venv
1. **Create a virtual environment**:

    ``` console
    python -m venv venv
    ```
    This requires Python to already be installed on your system. You could use the Python installed by the OS, by MiniForge, or any other approach.

2. **Activate the virtual environment**:

    === "macOS/Linux"
        ``` console
        source venv/bin/activate
        ```

    === "Windows"

        ``` console
        .\venv\Scripts\activate.bat
        ```

## Installing Canari-ML from Git

### Install latest default branch directly using pip

``` console
pip install git+https://github.com/CANARI-ML/canari-ml@main
```

### (Optional) Create local clone for development

1. **Clone the repository**:

    ``` console
    git clone git@github.com:CANARI-ML/canari-ml.git
    cd canari-ml
    ```

    To specify the branch to clone:

    ``` console
    git clone -b <branch_name> git@github.com:CANARI-ML/canari-ml.git
    cd canari-ml
    ```

2. **Install in development mode** (editable installation):

    ``` console
    pip install -e .
    ```

    To install for local development, including documentation:

    ``` console
    pip install -e .[dev,docs]
    ```

3. **Optional: For development, set up Git pre-commit hooks**:

    Run `pre-commit` once to set up the hook to run each time a commit is attempted:

    ``` console
    pre-commit install
    ```

## Next Steps

After setting up your environment, you can proceed with [downloading source ERA5 data](download.md).
