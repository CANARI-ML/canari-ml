# Advanced Usage


## Using Custom Loggers

By default, Tensorboard logging is enabled, however, you can easily switch to [WandB](http://wandb.ai/) (W&B) (after registering and logging in on your system) via a command line override.

### Login to W&B

To set-up W&B login credentials on your system, run the following and enter the API Key ([find your API Key here](https://docs.wandb.ai/support/find_api_key/)):

=== "Command"

    ``` console
    $ wandb login
    ```

=== "Output"

    ``` console
    wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
    wandb: You can find your API key in your browser here: https://wandb.ai/authorize?ref=models
    wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit:
    wandb: No netrc file found, creating one.
    wandb: Appending key for api.wandb.ai to your netrc file: /home/users/{USERNAME}/.netrc
    wandb: Currently logged in as: {USER_NAME} ({WANDB_USER}) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
    ```

This will store your API key locally under:

``` bash
~/.netrc
```

### Use W&B Logger

Now, the W&B logger can be used for training.

``` console
$ canari_ml train -cd configs/train/ -cn custom_train.yaml +logger=wandb
```

Alternatively, you could also define this in your training YAML config file as a config group after loading the default `/train` config file.

``` yaml
# @package _global_

defaults:
  - ../preprocess/train_1979_ua700_3days.yaml
  - /train
  - logger: wandb   # (1)!
  - _self_
```

1. Must be placed after loading the default `/train`, but any order after it.

## Specify Accelerators

By default, the "auto" keyword is used to allow PyTorch Lightning to auto-select the accelerator, and a single accelerator device is set to be used. To override which accelerator to use, and how many GPUs (or which GPU by GPU id):

``` console
canari_ml train -cd configs/train/ -cn custom_train.yaml trainer.accelerator="gpu" trainer.devices="'3'"
```

Refer to the [Lightning docs](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html#train-on-gpus) for more details.
