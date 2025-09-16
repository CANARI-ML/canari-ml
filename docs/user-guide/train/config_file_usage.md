# Override Defaults via Config File

To utilise a YAML config file instead of CLI overriding, you can following the examples in this section. And, as always, to understand the defaults and what can be overridden, use [`canari_ml train --help`](../../help/train.md).

## Examples

Below is an example configuration:

``` yaml title="configs/train/custom_train.yaml" linenums="1"
# @package _global_

defaults:                                   # (1)!
  - ../preprocess/train_demo_dataset.yaml   # (2)!
  - /train                                  # (3)!
  - _self_                                  # (4)!

train:
  dataset: preprocessed_data/train_demo_dataset/03_cache_demo_dataset/cached.DAY.north.json
  name: demo_train
  seed: 42
  epochs: 2
  workers: 4
  batch_size: 4
  shuffling: true
  wandb_group: demo_unet
  wandb_project: CANARI_Training

model:
  model_name: unet
  network:
    filter_size: 3
    n_filters_factor: 0.1
  litmodule:
    criterion:
      loss_type: mse
```

1. Always define `defaults` in the header of your custom config file.
2. Define path to the config file used for training.
3. Define `/train` to inherit from the default configuration.
4. Override the above defaults with values from this file. (The order matters, `_self_` should be defined last to override previous configs in this list).

You can run training with this config file using:

``` console
canari_ml train -cd configs/train/ -cn custom_train.yaml
```
