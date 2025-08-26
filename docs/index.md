# Canari-ML: North-Atlantic zonal wind forecasting codebase

> [!WARNING]
> This is a highly experimental codebase with constant changes with every development release, and is not ready for production use.

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
```bash
pip install git+https://github.com/CANARI-ML/canari-ml@main
```

2. **Usage**:
Run the following command to see available entry points:
```bash
canari_ml --help
```

This codebase uses [Hydra](https://hydra.cc/docs/intro/) for configuring different options, and following the quick-start guide will be helpful. It enables the user to change default options for download/preprocess/train/predict/postprocess/plot via either command line overrides, or via yaml config files (in a highly configurable and reproducibile manner).

This documentation will provide detailed guides on configuring models, preprocessing data, postprocessing and visualising the prediction results.

We hope you find Canari-ML useful! Let's get started with the [installation guide](installation.md).
