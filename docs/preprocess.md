# Preprocessing Data

## Overview

This section guides you through the use of the preprocessing command the generate the necessary dataset for training and prediction. You will learn how to use the `canari_ml preprocess` command to preprocess the source ERA5 data in such a manner that it is ready for ingestion into the ML model. And, like in previous sections, you can override default settings via command-line arguments or YAML configuration files, or both.

---

## Getting Started

### Prerequisites

For a training dataset:

- Follow all previous steps to [download](download.md) source data for all required variables, ready for preprocessing.
- This should include data for time before and after the dates being used for training/prediction because the model utilises historical data (defined by `lag_length`) to predict future steps.

For a prediction dataset:

- Ensure a training dataset has already been generated.
  - The normalisation parameters from this training dataset is used to normalise the prediction dataset against.
- A [trained](train.md) model that you want to generate predictions against.
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

``` console
canari_ml preprocess --help
```

### Training Subcommand

The `train` subcommand applies the [preprocessing steps](#usage), then processes the normalised data to generate a training dataset to Zarr format and a corresponding JSON config file, ready for GPU training.

#### Basic Usage

``` console
canari_ml preprocess train --help
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

``` console
canari_ml preprocess predict --help
```

This displays the help menu for the prediction command.

#### Running the Prediction Process

To execute the prediction preprocessing using the defaults:

``` console
canari_ml preprocess predict
```

---

## Override Defaults via Command-Line

Akin to the [download](download.md) approach, you can override the default behaviours using either command line overrides or a YAML configuration file, or even both.

If you are a fan of command line options, you will balk at the idea of using config files, and instead will like having an eternal list of options to override specific parameters like the following examples.

### Examples

#### Override Name and Forecast Steps for Training

To change the dataset name, and forecast length to 5 days:

``` console
canari_ml preprocess train input.name=primo input.forecast_length=5
```

By default, the number of historical days used to generate the dataset will match the forecast length, to adjust it to a specific number of days, you can override the `lag_length` parameter:

``` console
canari_ml preprocess train input.name=primo input.forecast_length=3 input.lag_length=3
```

#### Override More Options

To set the variables, dates and more options you want to use for training:

``` console
canari_ml preprocess train input.name=primo input.forecast_length=3 input.lag_length=2 input.vars.absolute="[ua500, ua700, va700]" input.vars.anomaly="[zg500,zg700]" input.dates.train.start="[1979-01-05, 1979-01-20]" input.dates.train.end="[1979-01-15, 1979-01-25]" preprocess_cache.output_batch_size=4 workers=2
```

- The variable names are a combination of [variables](download.md#variables) and [pressure-levels](download.md#pressure-levels) (if not a surface variable).
- `preprocess_cache.output_batch_size` defines the batch size for the dataset, this should ideally match the batch size used to run the training on your system.
- `workers` defines the number of concurrent threads/processes to use for the preprocessing steps. If running against a large number of variables and dates and running into `out of memory` or `segfault` errors, try reducing this number.

## Override Defaults via Config File

For more complex configurations, you will probably prefer using a YAML config file to drive the preprocessor.

#### Creating a Custom Config File

Create a custom configuration file `configs/preprocess/train_demo_dataset.yaml`:

``` yaml title="configs/preprocess/train_demo_dataset.yaml" linenums="1"
defaults:       # (1)!
  - /preprocess # (2)!
  - _self_      # (3)!

input:
  name: train_demo_dataset
  forecast_length: 3
  lag_length: 2
  vars:
    absolute:
      - ua500
      - ua700
    anomaly:    # (4)!
      - zg500
      - zg700
  dates:
    train:
      start:
        - 1979-01-05
        - 1979-01-20
      end:
        - 1979-01-15
        - 1979-01-25
    val:
      start:
        - 1979-01-25
      end:
        - 1979-01-25
    test:
      start:
        - 1979-01-26
      end:
        - 1979-01-26

preprocess_cache:
  output_batch_size: 4

workers: 2
```

1. Always define defaults in the header of your custom config file.
2. Uses the default preprocess config within the canari-ml codebase as base config.
3. Override the above defaults with values from this file. (The order matters, `_self_` should be defined last to override previous configs in this list).
4. If you do not want anomaly variables, you can set `anomaly: null`.

You can now run the preprocess command and point to this custom config file (just like in the [download](download.md#example-override-config-file) section).

``` console
canari_ml preprocess train -cd configs/preprocess/ -cn train_demo_dataset
```

---

## Create a Prediction Dataset

???+ note
    This step is only needed after a trained model is generated or made available.

The approach to creating a prediction dataset is very similar to creating a training dataset. The main difference is that it needs to use the same normalisation parameters as the training dataset used to train the model. And, there is no cached Zarr dataset generated since it would not be worth it for the prediction step.

### Using config file

``` yaml title="configs/preprocess/predict_trial_dataset.yaml" linenums="1"
defaults:
  - /preprocess
  - _self_

input:
  name: predict_trial_dataset
  forecast_length: 3
  lag_length: 2
  vars:
    absolute:
      - ua500
      - ua700
    anomaly:
      - zg500
      - zg700
  dates:
    predict:
      start:
        - 1979-01-26
      end:
        - 1979-01-26

preprocess_cache:
  output_batch_size: 4

workers: 2
```

To generate the prediction dataset, run:

``` console
canari_ml preprocess predict -cd configs/preprocess/ -cn predict_trial_dataset
```

???+ todo
    Define output structure

---

## Next Steps

After generating the necessary dataset cache, you can proceed with [creating a trained model](train.md).
