# Override Defaults via CLI

You can follow the [hydra documentation](https://hydra.cc/docs/advanced/override_grammar/basic/) to override any configuration option directly from the command line using the format `key=value`.

Due to the scope of this project, I would recommend not changing the following variables:

- `frequency`
- `output_group_by`
- `hemisphere`

### Examples

???+ warning
    Be wary that the default options mean that you will be downloading all data from 1979 to 2024, which can take a while!
    And, any of example commands might take a while to run, if only testing, I would recommend setting the date ranges to no more than a few days, and limit the number of variables & levels to download.

#### Select Specific Variables and Levels

``` bash
canari_ml download vars="[ua, zg]" levels="[700, 700]"
```

#### Custom Date Range

``` bash
canari_ml download dates.start="1980-01-01" dates.end="1990-12-31"
```

#### Common Override Options

``` bash
canari_ml download vars="[ua]" levels="[700]" dates.start="[1979-01-01]" dates.end="[1979-01-02]" delete_cache=true
```
