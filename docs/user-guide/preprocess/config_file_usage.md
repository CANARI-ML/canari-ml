# Override Defaults via Config File

For more complex configurations, you will probably prefer using a YAML config file to drive the preprocessor.

## Examples

### Creating a Custom Training Dataset

Create a custom configuration file `configs/preprocess/train_demo_dataset.yaml`:

``` yaml title="configs/preprocess/train_demo_dataset.yaml" linenums="1"
# @package _global_

defaults:       # (1)!
  - /preprocess # (2)!
  - _self_      # (3)!

input:
  name: demo_dataset
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

You can now run the preprocess command and point to this custom config file (just like in the [download](../download/index.md#example-override-config-file) section).

``` console
$ canari_ml preprocess train -cd configs/preprocess/ -cn train_demo_dataset
```

---

### Create a Prediction Dataset

???+ note
    This step is only needed after a trained model is generated or made available.

The approach to creating a prediction dataset is very similar to creating a training dataset. The main difference is that it needs to use the same normalisation parameters as the training dataset used to train the model. And, there is no cached Zarr dataset generated since it would not be worth it for the prediction step.

### Using config file

``` yaml title="configs/preprocess/predict_trial_dataset.yaml" linenums="1"
# @package _global_

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
$ canari_ml preprocess predict -cd configs/preprocess/ -cn predict_trial_dataset
```
