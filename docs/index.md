---
hide:
  - navigation
  - toc
---

# Canari-ML: North-Atlantic zonal wind forecasting codebase

<figure markdown="span">
  ![CANARI Image](assets/images/canari-hero-image.png){ width="300" }
  <figcaption>CANARI-ML</figcaption>
</figure>

???+ warning
    This is a highly experimental codebase with constant changes with every development release, and is not ready for production use.

Welcome to the documentation for Canari-ML, a machine learning library built with PyTorch Lightning for wind forecasting (zonal wind at 700hPa) across the North Atlantic.

## What is Canari-ML?

Canari-ML provides tools and models for processing environmental data and making wind forecast predictions. It is designed to work alongside the [environmental-forecasting initiative](http://github.com/environmental-forecasting/) for data preparation and preprocessing.

Canari-ML offers several key features:

- **Comprehensive Preprocessing**: Tools for loading, reprojecting, preparing and caching ERA5 datasets for ML training.
- **Integrated Experiment Tracking**: Track experiments using either Tensorboard, or WandB integration.
- **Prediction Capabilities**: Functions to train and predict on new data.
- **Visualisation Tools**: For analysing model results and training performance.

## Quick start

To begin using Canari-ML:

1. **Installation**:

``` console
pip install git+https://github.com/CANARI-ML/canari-ml@main
```

2. **Usage**:
Run the following command to see available entry points:

``` console
canari_ml --help
```


This documentation provides a foundation for configuring and running the training process. For more details on specific configuration options, refer to the [Hydra documentation](https://hydra.cc/docs/).

This codebase uses [Hydra](https://hydra.cc/docs/intro/) for configuring different options. For more details on specific configuration options. It enables the user to change default options for download/preprocess/train/predict/postprocess/plot via either command line overrides, or via yaml config files (in a highly configurable and reproducibile manner), or even both.

This documentation will provide detailed guides on configuring models, preprocessing data, postprocessing and visualising the prediction results.

We hope you find Canari-ML useful! Let's get started with the [installation guide](user-guide/getting-started/installation.md).

## Contributors

<a href="https://github.com/canari-ml/canari-ml/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=canari-ml/canari-ml" />
</a>
