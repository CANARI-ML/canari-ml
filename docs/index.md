---
hide:
  - navigation
  - toc
---

{%
    include-markdown "../README.md"
    start="<!--header-start-->"
    end="<!--header-end-->"
%}

Welcome to the documentation for Canari-ML, a machine learning library built with PyTorch Lightning for wind forecasting (zonal wind at 700hPa) across the North Atlantic.

This documentation provides a foundation for configuring and running the training process. For more details on specific configuration options, refer to the [Hydra documentation](https://hydra.cc/docs/).

This codebase uses [Hydra](https://hydra.cc/docs/intro/) for configuring different options. For more details on specific configuration options. It enables the user to change default options for download/preprocess/train/predict/postprocess/plot via either command line overrides, or via yaml config files (in a highly configurable and reproducibile manner), or even both.

This documentation will provide detailed guides on configuring models, preprocessing data, postprocessing and visualising the prediction results.

{%
    include-markdown "../README.md"
    start="<!--main-start-->"
    end="<!--main-end-->"
%}


We hope you find Canari-ML useful! Let's get started with the [installation guide](user-guide/getting-started/installation.md).

{%
    include-markdown "../README.md"
    start="<!--contributors-start-->"
    end="<!--contributors-end-->"
%}
