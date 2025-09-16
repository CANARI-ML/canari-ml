# Override Defaults via Config File

For more complex customisations, you may prefer using a YAML config file with the non-defaults you want to set. This can be set up in many different ways following Hydra's approach, however, I will provide an example layout below.

I will store the custom config within a `configs/download` directory.

## Example Override Config File

``` yaml title="configs/download/small_download.yaml" linenums="1"
defaults:     # (1)!
  - /download # (2)!
  - _self_    # (3)!

vars:
  - ua
  - zg
  - tas
  - tos
  - sic

levels:
  - 500|700   # (4)!
  - 500|700
  - null      # (5)!
  - null
  - null

dates:
  start: 1979-01-01
  end: 1979-01-31
```

1. Always define defaults in the header of your custom config file.
2. Uses the default download config within the canari-ml codebase as base config.
3. Override the above defaults with values from this file. (The order matters, `_self_` should be defined last to override previous configs in this list).
4. Download multiple pressure levels, separated by the `|` operator.
5. Surface-level variables do not have a pressure-level associated with them, so they are set to `null`. Check [Variables](index.md#variables) section on whether a variable is surface-level or has multiple pressure-levels, in which case, you should specify the pressure levels you want to download.

???+ note
    The AWS ERA5 mirror stores all 37 pressure levels in a single netCDF file (Initial version of this downloader extracted just the necessary variables from the S3 bucket using boto and xarray, but, this was slower than downloading the entire file on BAS HPC, so, code now downloads the entire file, then extracts the required pressure levels).

    If you want to download multiple pressure levels, it is recommended to set to download all of them in one go since it will download the entire 37 pressure level file no matter how many pressure levels you specify, so, this mean not having to redownload the entire file further down the line.

You can now run the download command and point to this config file.

``` console
$ canari_ml download --config-dir configs/download/ --config-name small_download
```

or, for brevity, you can use the short options:

``` console
$ canari_ml download -cd configs/download/ -cn small_download
```

You can confirm the default options have been overridden by adding the `--help` flag:

``` console
$ canari_ml download -cd configs/download/ -cn small_download --help
```

where the following hydra options are,

* `--config-dir` or `-cd`: The directory where your custom config file is located
* `--config-name` or `-cn`: The name of your custom config file, either with or without the `.yaml` extension

The default location that Hydra looks for config files is set to where the default config files are installed using pip, or where your cloned repo is.

# Combine CLI and Config File Overrides

You can override settings in the config file while providing additional CLI arguments. CLI overrides take precedence over the config file.

## Example Command

``` console
$ canari_ml download -cd configs/download/ -cn small_download vars="[ua]" levels="[700]" dates.start="1979-01-01"
```

This would override the definition in the custom config file and only download `ua` at `700hPa`, with dates between `1979-01-01` to `1979-01-31`.

And, once again, you can verify the default options have been overridden by adding the `--help` flag:

``` console
$ canari_ml download -cd configs/download/ -cn small_download vars="[ua]" levels="[700]" dates.start="1979-01-01" --help
```
