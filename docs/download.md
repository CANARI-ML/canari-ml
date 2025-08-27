# Download ERA5 Data

Canari-ML provides a flexible configuration system using Hydra to download ERA5 reanalysis data. This guide explains how to use the `canari_ml download` command, including overriding default settings via CLI arguments or custom config files.

## Default Configuration

The main command for downloading data is `canari_ml download`. To find the default options, and what configuration options can be changed, run:

``` console
canari_ml download --help
```

## General config options

| Name                | Type      | Default Value | Description                                                              |
|---------------------|-----------|---------------|--------------------------------------------------------------------------|
| frequency           | str       | `DAY`         | The temporal resolution of the data to download.                         |
| output_group_by     | str       | `YEAR`        | How output files are grouped (e.g., by `YEAR` or `MONTH`).               |
| hemisphere          | str       | `north`       | Which hemisphere to download data for (`north` or `south`).              |
| workers             | int       | 4             | Number of parallel workers for downloading.                              |
| delete_cache        | bool      | false         | Whether to delete cached files after processing.                         |
| cache_only          | bool      | false         | Only download and cache data without processing.                         |
| overwrite_config    | bool      | true          | Overwrite existing configuration files during setup.                     |

## Variables

Variables available for download:

| CMIP6 variable name | Description                     | ECMWF ID | ECMWF Short Name | Dataset               | Comments                                                                      |
|---------------------|---------------------------------|----------|------------------|-----------------------|-------------------------------------------------------------------------------|
| ta                  | Air temperature                 | 130      | t                | pressure-level        | Temperature in the atmosphere (K)                                             |
| ua                  | Zonal wind component            | 131      | u                | pressure-level        | Eastward component of the wind (m/s)                                          |
| va                  | Meridional wind component       | 132      | v                | pressure-level        | Horizontal speed of air moving towards the north (m/s)                        |
| zg                  | Geopotential height             | 129      | z                | pressure-level        | This downloads geopotential {z}, which is converted during the dataset preprocess step to geopotential height {zg} |
| hus                 | Specific humidity               | 133      | q                | pressure-level        | Mass of water vapour per kilogram of moist air (kg kg-1)                      |
| ps                  | Surface pressure                | 134      | sp               | surface-level         | Pressure (force per unit area) of the atmosphere on the surface of land, sea and in-land water (Pa) |
| psl                 | Sea level pressure              | 151      | msl              | surface-level         | Pressure (force per unit area) of the atmosphere adjusted to the height of mean sea level (Pa) |
| sic                 | Sea ice concentration           | 262001   | ci               | surface-level         | Fraction of a grid box which is covered by sea ice (1)                        |
| tas                 | Near-surface air temperature    | 167      | 2t               | surface-level         | Temperature of air at 2m above the surface of land, sea or in-land waters (K) |
| tos                 | Sea Surface Temperature         | 34       | sstk             | surface-level         | Temperature of sea water near the surface (K)                                 |

## Levels

Pressure levels for variables:

| Name    | Type      | Default Value                      | Description                                   |
|---------|-----------|------------------------------------|-----------------------------------------------|
| levels  | list[str] | `[2|10|50|100|250|500|700]`        | Pressure levels for the selected variables.   |

## Dates

Date range configuration:

| Name      | Type      | Default Value           | Description                           |
|-----------|-----------|-------------------------|---------------------------------------|
| start     | str       | `1979-01-01`            | Start date of data download.          |
| end       | str       | `2024-12-31`            | End date of data download.            |

This configuration structure allows you to customize your data downloading and processing workflow. Use these tables as a reference when setting up your `config.yml` file.


## Override Defaults via CLI

You can follow the [hydra documentation](https://hydra.cc/docs/advanced/override_grammar/basic/) to override any configuration option directly from the command line using the format `key=value`.

Due to the scope of this project, I would recommend not changing the following variables:

- `frequency`
- `output_group_by`
- `hemisphere`

### Example 1: Select Specific Variables and Levels

???+ warning
    Be wary that the default options mean that you will be downloading all data from 1979 to 2024, which can take a while!

``` console
canari_ml download vars="[ua, va]" levels="[700, 700]"
```

### Example 2: Custom Date Range

``` console
canari_ml download dates.start="1980-01-01" dates.end="1990-12-31"
```

### Example 3: Common Override Options

``` console
canari_ml download vars="[ua]" levels="[700]" dates.start="[1979-01-01]" dates.end="[1979-01-02]" delete_cache=true
```

## Override Defaults via Config File

For more complex customisations, you may prefer create a YAML config file and using the `--config-file` option.

### Example Override Config File

``` yaml title="custom_download.yaml"
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

The surface-level variables do not have a level associated with them, so they are set to `null`. Check [Variables](#variables) section on whether a variable is surface-level or has multiple pressure-levels.

### Run with Custom Config

``` console
canari_ml download --config-file custom_download.yaml
```

## Combine CLI and Config File Overrides

You can override settings in the config file while providing additional CLI arguments. CLI overrides take precedence over the config file.

### Example Command

``` console
canari_ml download --config-file custom_download.yaml vars="[ua]" levels="[700]" dates.start="1985-01-01"
```

This would override the definition in the config file and only download `ua` at `700hPa`, with dates between `1985-01-01` to `1990-12-31`.

## Summary

The `canari_ml download` command offers extensive flexibility through Hydra's configuration system. You can:

1. Override defaults directly via CLI arguments.
2. Create custom JSON config files for complex settings.
3. Combine CLI and config file overrides, with CLI taking precedence.

This allows you to tailor the download process for the variables you want to train with.
