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


## Summary

The `canari_ml download` command offers extensive flexibility through Hydra's configuration system. You can:

1. Override defaults directly via CLI arguments.
2. Use custom YAML config files to define complex overrides of the default behaviour.
3. Combine CLI and config file overrides, with CLI taking precedence.

This allows you to tailor the download process for the variables you want to train with.

## Next Steps

After downloading the necessary data, you can proceed with [preprocessing this data to get it ready for training](../preprocess/index.md).
