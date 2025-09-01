# Basics of Using Hydra Configuration

Hydra is a framework for managing hyperparameters and configurations in a reproducible manner. It provides an elegant way to manage complex configurations while keeping the codebase clean.

## Overview

Hydra enables dynamic configuration through a hierarchical approach which allows overriding of values directly via the command line interface (CLI), or via YAML config files, making it flexible and easy to adapt. For more detailed usage guidance, follow the [Hydra documentation](https://hydra.cc/docs/intro). This page only covers the most basic of usages.

## Command Line Interface (CLI) Overriding

One of the most powerful features of Hydra is its ability to override configurations via the command line.

To view the different available options, use the `--help` command, as an example of the training command:

``` console
canari_ml train --help
```

In any command, where the default is set to `???`, this is a parameter where the user **must** define it themselves, and there is no default value.

This documentation captures the latest help outputs for reference, and can be found under the `CLI help` sub-section of each section in `Main Usage`.

### Basic CLI Override

To override the default parameters at runtime:

``` console
canari_ml train model.network.filter_size=5 model.litmodule.criterion.learning_rate=0.001
```

This will override the default values of `model.network.filter_size` and `model.litmodule.criterion.learning_rate` in your configuration.

### Adding New Values

You can even add new values that were not present in the original configuration file:

``` console
canari_ml train +new_param="extra_value"
```

This will add new_param to the configuration dynamically and allow you to access it within the Python code.

## Summary

Hydra provides a robust and flexible way to manage configurations in your Python projects. By leveraging type safety, configuration injection, and command line overrides, you can maintain clean code while efficiently managing hyperparameters and settings.

For more details, check the [Hydra documentation](https://hydra.cc/docs/intro), including the sections on multi-runs with sweeps which is used for ensemble runs and submitting jobs on SLURM.
