<!--header-start-->

<figure markdown="span" align="center">
  <img src="docs/assets/images/canari-hero-image.png" alt="Canari-ML Image" width="300" height="254">
  <figcaption>CANARI-ML</figcaption>
</figure>

<div align="center">
  <h1 align="center" style="display:inline-block;">Canari-ML: North-Atlantic zonal wind forecasting codebase</h1>
</div>


<p align="center">
  <a href="https://github.com/canari-ml/canari-ml/actions/workflows/test.yaml?query=branch%3Amain">
    <img src="https://github.com/canari-ml/canari-ml/actions/workflows/test.yaml/badge.svg?branch=main" alt="Testing">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  </a>
  <a href="https://canari-ml.readthedocs.io/">
    <img src="https://img.shields.io/badge/docs-canari--ml.io-green" alt="Docs">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Formatted with ruff">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python Version">
  </a>
</p>

<p align="center">
  <img alt="GitHub Pull Requests Open" src="https://img.shields.io/github/issues-pr/canari-ml/canari-ml">
  <img alt="GitHub Pull Requests Closed" src="https://img.shields.io/github/issues-pr-closed/canari-ml/canari-ml">
  <img alt="GitHub Issues Open" src="https://img.shields.io/github/issues-raw/canari-ml/canari-ml">
  <img alt="GitHub Issues Closed" src="https://img.shields.io/github/issues-closed/canari-ml/canari-ml">
</p>

<p align="center">
  <img alt="Weights and Biases Logo" src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white">
  <img alt="PyTorch Logo" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="Linux Logo" src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black">
</p>

???+ warning
    This is a highly experimental codebase with constant changes with every development release, and is not ready for production use.

Canari-ML is a machine learning library built with PyTorch Lightning for wind forecasting (zonal wind at 700hPa) across the North Atlantic.

<!--header-end-->

<!--main-start-->

## What is Canari-ML?

Canari-ML provides tools and models for processing environmental data and making wind forecast predictions. It is designed to be used in conjunction with the [environmental-forecasting initiative](http://github.com/environmental-forecasting/) which is used for data download and for majority of the pre-processing steps to prepare the source data for training and prediction.

## Features

- **Models**: Currently, a reference UNet model is implemented for wind forecasting.
- **Preprocessing**: Utilities for loading, reprojecting, preparing and caching ERA5 datasets for ML training.
- **Integrated Experiment Tracking**: Track experiments using either Tensorboard, or WandB integration.
- **Prediction**: Functions to train and predict on new data.
- **Visualisation**: Tools for visualising the results of predictions and model training.

## Quick start

To begin using Canari-ML:

1. **Installation**:

``` bash
pip install git+https://github.com/CANARI-ML/canari-ml@main
```

2. **Usage**:
Run the following command to see available entry points:

``` bash
canari_ml --help
```

## License

CANARI-ML is licensed under the MIT license. See [LICENSE](https://github.com/CANARI-ML/canari-ml/blob/main/LICENSE) for more information.

<!--main-end-->

## Documentation

The latest documentation can be found on [Read the Docs](https://canari-ml.readthedocs.io).


## Contributing

Contributions are welcome!

Please follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard for commit messages. Any that do not follow this standard will not be merged into the main branch and may be rejected.

Please see [CONTRIBUTING](https://github.com/CANARI-ML/canari-ml/blob/main/CONTRIBUTING.md) for more information on how to contribute.

CANARI-ML is a work in progress and will be updated as development progresses.

## Release

This repo uses the [Commitizen](https://commitizen-tools.github.io/commitizen/) package (installed as dev dependency) to manage changelogs and package version control.

To bump to the next stable version:

```bash
cz bump
```

Examples:

v0.0.1 → v0.0.2

To release an alpha version:

```bash
cz bump --prerelease alpha
```

Examples:

v1.0.4 → v1.0.5-alpha.0
v1.0.5-alpha.0 → v1.0.5-alpha.1

To start a new patch-level prerelease explicitly:

```bash
cz bump --increment patch --prerelease alpha
```

Examples:

v1.2.4-alpha.3 → v1.2.5-alpha.0

For full documentation on its usage, please peruse the [Commitizen docs](https://commitizen-tools.github.io/commitizen/commands/bump/).


<!--contributors-start-->

## Contributors

<a href="https://github.com/canari-ml/canari-ml/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=canari-ml/canari-ml" />
</a>

<!--contributors-end-->
