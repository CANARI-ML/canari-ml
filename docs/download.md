# Download ERA5 Data

Canari-ML provides a flexible configuration system using Hydra to download ERA5 reanalysis data. This guide explains how to use the `canari_ml download` command, including overriding default settings via CLI arguments or custom config files.

## Default Configuration

The main command for downloading data is `canari_ml download`. To find the default options, and what configuration options can be changed, run:

```console
canari_ml download --help
```

- **frequency**: `DAY` (Downloads daily data)
- **output_group_by**: `YEAR` (Groups output files by year)
- **hemisphere**: `north` (Downloads data for the northern hemisphere)
- **workers**: `4` (Number of workers for parallel downloading)
- **delete_cache**: `false` (Does not delete cached data, something to consider if you're running out of space)
- **cache_only**: `false` (Only downloads the source data into `data/aws/cache/`, does not process them)
- **overwrite_config**: `true` (Overwrites existing config files)
- **vars**: (Variables to download)
    - zg
    - ua
    - va
    - tos
    - tas
    - sic
- **levels**: `[2|10|50|100|250|500|700]` (Corresponding levels for each variable)
- **dates**:
    - **start**: `'1979-01-01'`
    - **end**: `'2024-12-31'`

## Override Defaults via CLI

You can follow the [hydra documentation](https://hydra.cc/docs/advanced/override_grammar/basic/) to override any configuration option directly from the command line using the format `key=value`.

Due to the scope of this project, I would recommend not changing the following variables:

- `frequency`
- `output_group_by`
- `hemisphere`

### Example 1: Select Specific Variables and Levels

???+ warning
    Be wary that the default options mean that you will be downloading all data from 1979 to 2024, which can take a while!

```console
canari_ml download vars="[ua, va]" levels="[700, 700]"
```

### Example 2: Custom Date Range

```console
canari_ml download dates.start="1980-01-01" dates.end="1990-12-31"
```

### Example 3: Common Override Options

```console
canari_ml download vars="[ua]" levels="[700]" dates.start="[1979-01-01]" dates.end="[1979-01-02]" delete_cache=true
```

## Override Defaults via Config File

For more complex customisations, you may prefer create a YAML config file and using the `--config-file` option.

### Example Config File (`custom_download.yaml`)

``` yaml
vars:
  - ua
  - va
  - tos
  - tos
  - sic

levels:
  - 250|500|700
  - 500|700
  - null
  - null
  - null

dates:
  start: 1980-01-01
  end: 1990-12-31
```

### Run with Custom Config

```console
canari_ml download --config-file custom_download.yaml
```

## Combine CLI and Config File Overrides

You can override settings in the config file while providing additional CLI arguments. CLI overrides take precedence over the config file.

### Example Command

```console
canari_ml download --config-file custom_download.yaml vars="[ua]" levels="[700]" dates.start="1985-01-01"
```

This would override the definition in the config file and only download `ua` at `700hPa`, with dates between `1985-01-01` to `1990-12-31`.

## Summary

The `canari_ml download` command offers extensive flexibility through Hydra's configuration system. You can:

1. Override defaults directly via CLI arguments.
2. Create custom JSON config files for complex settings.
3. Combine CLI and config file overrides, with CLI taking precedence.

This allows you to tailor the download process for the variables you want to train with.
