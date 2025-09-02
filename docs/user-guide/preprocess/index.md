# Preprocessing Data

## Overview

This section guides you through the use of the preprocessing command the generate the necessary dataset for training and prediction. You will learn how to use the `canari_ml preprocess` command to preprocess the source ERA5 data in such a manner that it is ready for ingestion into the ML model. And, like in previous sections, you can override default settings via command-line arguments or YAML configuration files, or both.

---

## Getting Started

### Prerequisites

For a training dataset:

- Follow all previous steps to [download](../download/index.md) source data for all required variables, ready for preprocessing.
- This should include data for time before and after the dates being used for training/prediction because the model utilises historical data (defined by `lag_length`) to predict future steps.

For a prediction dataset:

- Ensure a training dataset has already been generated.
  - The normalisation parameters from this training dataset is used to normalise the prediction dataset against.
- A [trained](../train/index.md) model that you want to generate predictions against.
  - The trained model symlinks to the location of the training dataset used to create the trained model which the code uses to figure out what the normalisation parameters were.

---

## Usage

The `canari_ml preprocess` command preprocesses the ERA5 data running the following steps:

- Create train/val/predict data splits across input date ranges (either the defaults, or user-specified overrides).
- Reproject the data from source CRS of `EPSG:4326` to `EPSG:6931` (by default, but is configurable).
- Normalise the dataset to transform the range of variables to a standard scale.
  - Will ensure that each variable contributes proportionally to the model's learning process, else, certain variables may have higher/lower weighting based on their range.
  - If geopotential (`z`) is defined, convert it to geopotential height (`zg`).
- Apply hemisphere mask to mask out the region below 0&deg; latitude (i.e., masking out the Southern hemisphere).

To see the subcommands available, run:

``` console exec="on" source="tabbed-left" result="ansi" tabs="Command|Output"
$ canari_ml preprocess --help
```

### Training Subcommand

The `train` subcommand applies the [preprocessing steps](#usage), then processes the normalised data to generate a training dataset to Zarr format and a corresponding JSON config file, ready for GPU training.

#### Basic Usage


``` console exec="on" source="tabbed-left" result="ansi" tabs="Command|Output"
$ canari_ml preprocess train --help
```

This displays the help menu with all available default configuration options. And, will inform you of what options are available to override.

#### Running the Training Process

To execute the training preprocessing with the defaults:

``` console
canari_ml preprocess train
```

---

### Prediction Subcommand

The `predict` subcommand applies the [preprocessing steps](#usage), then just outputs the JSON config file without generating a cached dataset since there would not be of much performance benefit in trying to cache the dataset for prediction.

#### Basic Usage

``` console exec="on" source="tabbed-left" result="ansi" tabs="Command|Output"
$ canari_ml preprocess predict --help
```

This displays the help menu for the prediction command.

#### Running the Prediction Process

To execute the prediction preprocessing using the defaults:

``` console
$ canari_ml preprocess predict
```
