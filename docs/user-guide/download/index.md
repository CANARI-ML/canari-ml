# Download ERA5 Data

Canari-ML provides a flexible configuration system using Hydra to download ERA5 reanalysis data. This guide explains how to use the `canari_ml download` command, including overriding default settings via CLI arguments or custom config files.

## Default Configuration

The main command for downloading data is `canari_ml download`. To find the default options, and what configuration options can be changed, run:

``` console exec="on" source="tabbed-left" result="ansi" tabs="Command|Output"
$ canari_ml download --help
```

The source of this data is [NSF NCAR](https://rda.ucar.edu/datasets/d633000) which hosts a mirror of ERA5 dataset on [AWS S3](https://registry.opendata.aws/nsf-ncar-era5/) (amongst other access methods). This uses [download-toolbox](https://download-toolbox.readthedocs.io/en/latest/) from the [environmental-forecasting initiative](https://environmental-forecasting.github.io/).

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
| hus                 | Specific humidity               | 133      | q                | pressure-level        | Mass of water vapour per kilogram of moist air (kg kg-1)                      |
| ta                  | Air temperature                 | 130      | t                | pressure-level        | Temperature in the atmosphere (K)                                             |
| ua                  | Zonal wind component            | 131      | u                | pressure-level        | Eastward component of the wind (m/s)                                          |
| va                  | Meridional wind component       | 132      | v                | pressure-level        | Horizontal speed of air moving towards the north (m/s)                        |
| zg                  | Geopotential height             | 129      | z                | pressure-level        | This downloads geopotential {z}, which is converted during the dataset preprocess step to geopotential height {zg} |
| ps                  | Surface pressure                | 134      | sp               | surface-level         | Pressure (force per unit area) of the atmosphere on the surface of land, sea and in-land water (Pa) |
| psl                 | Sea level pressure              | 151      | msl              | surface-level         | Pressure (force per unit area) of the atmosphere adjusted to the height of mean sea level (Pa) |
| sic                 | Sea ice concentration           | 262001   | ci               | surface-level         | Fraction of a grid box which is covered by sea ice (1)                        |
| tas                 | Near-surface air temperature    | 167      | 2t               | surface-level         | Temperature of air at 2m above the surface of land, sea or in-land waters (K) |
| tos                 | Sea Surface Temperature         | 34       | sstk             | surface-level         | Temperature of sea water near the surface (K)                                 |

???+ note
    While there are additional variables that are available for download from the ERA5 AWS data mirror, the variables above are the only ones that have been mapped and downloadable through this interface currently.

## Pressure Levels

Pressure levels for variables:

| Name    | Type      | Default Value                      | Description                                   |
|---------|-----------|------------------------------------|-----------------------------------------------|
| levels  | list[str] | `[2|10|50|100|250|500|700]`        | Pressure levels for the selected variables.   |

For more information on what pressure levels are available, check out the [source data documentation](https://rda.ucar.edu/datasets/d633000/detailed_metadata/?view=level).

Your options for pressure levels are:

`[1|2|3|5|7|10|20|30|50|70|100|125|150|175|200|225|250|300|350|400|450|500|550|600|650|700|750|775|800|825|850|875|900|925|950|975|1000]`

## Dates

Configuring date ranges to download:

| Name            | Type      | Default Value           | Description                           |
|-----------------|-----------|-------------------------|---------------------------------------|
| dates.start     | str       | `1979-01-01`            | Start date of data download.          |
| dates.end       | str       | `2024-12-31`            | End date of data download.            |

This configuration structure allows you to customise your data downloading and processing workflow. Use these tables as a reference when setting up your `config.yml` file.


## Override Defaults via CLI

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

``` console
canari_ml download vars="[ua, va]" levels="[700, 700]"
```

#### Custom Date Range

``` console
canari_ml download dates.start="1980-01-01" dates.end="1990-12-31"
```

#### Common Override Options

``` console
canari_ml download vars="[ua]" levels="[700]" dates.start="[1979-01-01]" dates.end="[1979-01-02]" delete_cache=true
```

## Override Defaults via Config File

For more complex customisations, you may prefer using a YAML config file with the non-defaults you want to set. This can be set up in many different ways following Hydra's approach, however, I will provide an example layout below.

I will store the custom config within a `configs/download` directory.

### Example Override Config File

``` yaml title="configs/download/small_download.yaml" linenums="1"
defaults:     # (1)!
  - /download # (2)!
  - _self_    # (3)!

vars:
  - ua
  - zg
  - tos
  - tos
  - sic

levels:
  - 250|500|700 # (4)!
  - 500|700
  - null        # (5)!
  - null
  - null

dates:
  start: 1980-01-01
  end: 1990-12-31
```

1. Always define defaults in the header of your custom config file.
2. Uses the default download config within the canari-ml codebase as base config.
3. Override the above defaults with values from this file. (The order matters, `_self_` should be defined last to override previous configs in this list).
4. Download multiple pressure levels, separated by the `|` operator.
5. Surface-level variables do not have a pressure-level associated with them, so they are set to `null`. Check [Variables](#variables) section on whether a variable is surface-level or has multiple pressure-levels, in which case, you should specify the pressure levels you want to download.

???+ note
    The AWS ERA5 mirror stores all 36 pressure levels in a single netCDF file (Initial version of this downloader extracted just the necessary variables from the S3 bucket using boto and xarray, but, this was slower than downloading the entire file on BAS HPC, so, code now downloads the entire file, then extracts the required pressure levels).

    If you want to download multiple pressure levels, it is recommended to set to download all of them in one go since it will download the entire 36 pressure level file no matter how many pressure levels you specify, so, this mean not having to redownload the entire file further down the line.

You can now run the download command and point to this config file.

``` console
canari_ml download --config-dir configs/download/ --config-name small_download
```

or, for brevity, you can use the short options:

``` console
canari_ml download -cd configs/download/ -cn small_download
```

You can confirm the default options have been overridden by adding the `--help` flag:

``` console
canari_ml download -cd configs/download/ -cn small_download --help
```

where the following hydra options are,

* `--config-dir` or `-cd`: The directory where your custom config file is located
* `--config-name` or `-cn`: The name of your custom config file, either with or without the `.yaml` extension

The default location that Hydra looks for config files is set to where the default config files are installed using pip, or where your cloned repo is.

## Combine CLI and Config File Overrides

You can override settings in the config file while providing additional CLI arguments. CLI overrides take precedence over the config file.

### Example Command

``` console
canari_ml download -cd configs/download/ -cn small_download vars="[ua]" levels="[700]" dates.start="1985-01-01"
```

This would override the definition in the custom config file and only download `ua` at `700hPa`, with dates between `1985-01-01` to `1990-12-31`.

And, once again, you can verify the default options have been overridden by adding the `--help` flag:

``` console
canari_ml download -cd configs/download/ -cn small_download vars="[ua]" levels="[700]" dates.start="1985-01-01" --help
```

## Summary

The `canari_ml download` command offers extensive flexibility through Hydra's configuration system. You can:

1. Override defaults directly via CLI arguments.
2. Use custom YAML config files to define complex overrides of the default behaviour.
3. Combine CLI and config file overrides, with CLI taking precedence.

This allows you to tailor the download process for the variables you want to train with.


???+ todo
    Define output structure

## Next Steps

After downloading the necessary data, you can proceed with [preprocessing this data to get it ready for training](../preprocess/index.md).
