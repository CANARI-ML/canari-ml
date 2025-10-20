# Override Defaults via Config File

## Examples

### Example 1: Basic prediction config

### Generate Prediction Dataset

``` yaml title="configs/predict/custom_train/preprocess_1976_example.yaml" linenums="1"
# @package _global_

defaults:
  - ../../train/custom_train.yaml   # (1)!
  - ../../preprocess/train_demo_dataset.yaml   # (2)!
  - _self_

# To create prediction dataset
input:
  name: 1976_example
  dates:
    predict:
      start:
        - 1979-01-26    # (3)!
      end:
        - 1979-01-26    # (3)!

workers: 1
```

1. Point to the config of the training model (relative to where this config file is placed).
2. Point to the config of the training dataset (relative to where this config file is placed).
3. This should be a list of date ranges you would want to generate predictions for.

```bash
canari_ml preprocess predict -cd configs/predict/custom_train/ -cn preprocess_1976_example
```

### Generate Predictions

``` yaml title="configs/predict/custom_train/1979-01-26.yaml" linenums="1"
# @package _global_

defaults:
  - ../../train/custom_train.yaml   # (1)!
  - /predict
  - _self_

predict:
  name: 1979-01-26
  dates:
    - 1979-01-26     # (2)!
  dataset: preprocessed_data/predict_1976_example/03_cache_1976_example/cached.DAY.north.json # (3)!
  seed: 42
  workers: 4
  batch_size: 4
```

1. Relative path to the Hydra config used to generate the trained model (relative to the prediction config file).
2. You can define a list of dates for which to make predictions. This date has to exist in the prediction dataset being used.
3. Relative path to the prediction dataset config file (relative to the path you are running the `canari_ml predict` command from).

This can be run using:

```bash
canari_ml predict -cd configs/predict/custom_train/ -cn 1979-01-26
```

### Example 2: Combined prediction, postprocess and plotting config

You can also (optionally) enable the default `/postprocess` and `/plot` config files to the prediction config, which will allow you to use the same config file to generate the output netCDF, and plot the results.

``` yaml title="configs/predict/custom_train/1979-01-26_and_plot.yaml" linenums="1" hl_lines="6-7"
# @package _global_

defaults:
  - ../../train/custom_train.yaml
  - /predict
  - /postprocess: netcdf
  - /plot: ua700
  - _self_

predict:
  name: 1979-01-26
  dates:
    - 1979-01-26
  dataset: preprocessed_data/predict_1976_example/03_cache_1976_example/cached.DAY.north.json
  seed: 42
  workers: 4
  batch_size: 4
```

This can be run using:

```bash
canari_ml predict -cd configs/predict/custom_train/ -cn 1979-01-26_and_plot
```
