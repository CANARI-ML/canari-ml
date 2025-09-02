# Override Defaults via Command-Line

Akin to the [download](../download/index.md) approach, you can override the default behaviours using either command line overrides or a YAML configuration file, or even both.

If you are a fan of command line options, you will balk at the idea of using config files, and instead will like having an eternal list of options to override specific parameters like the following examples.

## Examples

### Override Name and Forecast Steps for Training

To change the dataset name, and forecast length to 5 days:

``` console
$ canari_ml preprocess train input.name=primo input.forecast_length=5
```

By default, the number of historical days used to generate the dataset will match the forecast length, to adjust it to a specific number of days, you can override the `lag_length` parameter:

``` console
$ canari_ml preprocess train input.name=primo input.forecast_length=3 input.lag_length=3
```

### Override More Options

To set the variables, dates and more options you want to use for training:

``` console
$ canari_ml preprocess train input.name=primo input.forecast_length=3 input.lag_length=2 input.vars.absolute="[ua500, ua700]" input.vars.anomaly="[zg500,zg700]" input.dates.train.start="[1979-01-05, 1979-01-20]" input.dates.train.end="[1979-01-15, 1979-01-25]" preprocess_cache.output_batch_size=4 workers=2
```

- The variable names are a combination of [variables](../download/index.md#variables) and [pressure-levels](../download/index.md#pressure-levels) (if not a surface variable).
- `preprocess_cache.output_batch_size` defines the batch size for the dataset, this should ideally match the batch size used to run the training on your system.
- `workers` defines the number of concurrent threads/processes to use for the preprocessing steps. If running against a large number of variables and dates and running into `out of memory` or `segfault` errors, try reducing this number.

## Next Steps

After generating the necessary dataset cache, you can proceed with [creating a trained model](../train/index.md).
