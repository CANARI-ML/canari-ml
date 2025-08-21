import argparse
import datetime as dt
import glob
import itertools
import logging
import os
from importlib.metadata import version
from pathlib import Path

import dask.array as da
import numpy as np
import orjson
import pandas as pd
import xarray as xr
from cf_units import Unit
from dateutil.relativedelta import relativedelta
from omegaconf import DictConfig
from preprocess_toolbox.interface import get_processor_from_source
from preprocess_toolbox.utils import get_config

import canari_ml
from canari_ml.data.dataloader import CANARIMLDataSetTorch
from canari_ml.data.masks.era5 import Masks
from canari_ml.models.networks.pytorch import CACHE_SYMLINK_DIR


def get_prediction_data(
    predict_dir_root: str, seeds: list[int], date: dt.date, return_ensemble_data: bool = False
) -> tuple | None:
    """
    Get prediction data from ensemble of numpy files for given date.

    Args:
        root: Root directory path to pipeline results.
        name: Name of the prediction.
        date: Forecast date to get prediction data for.
        return_ensemble_data (optional): Whether to also return full ensemble data
            array, or just the mean. Defaults to False.

    Returns:
        tuple:
            - If `return_ensemble_data` is True:
              Returns (data_mean, full_data_ensemble, number_of_ensemble_members)
            - If `return_ensemble_data` is False:
              Returns (data_mean, number_of_ensemble_members)
    """
    logging.info("Post-processing {}".format(date))

    np_files = []
    for seed in seeds:
        glob_str = os.path.join(
            predict_dir_root, str(seed), "raw_predictions", date.strftime("%Y_%m_%d.npy")
        )
        logging.info(f"Globbing prediction files for seed: {seed}")

        logging.debug(f"Glob string for prediction files:\n {glob_str}")

        np_files.append(glob.glob(glob_str))

    np_files = list(itertools.chain(*np_files))
    if not len(np_files):
        logging.warning("No files found")
        return None

    # n_channels is the number of prediction variables by the model
    # (ensemble, n_channels, xc, yc, leadtime)
    data = [np.load(f) for f in np_files]
    # Since only predicting one variable, squeeze n_channels dimension out
    data = np.asarray(data).squeeze(1)
    ens_members = data.shape[0]

    logging.info("Data read from disk: {} from: {}".format(data.shape, np_files))

    # mean:    (yc, xc, leadtime)
    # std_dev: (yc, xc, leadtime)
    data_mean, data_std = data.mean(axis=0), data.std(axis=0)

    if return_ensemble_data:
        return data_mean, data_std, data, ens_members
    else:
        return data_mean, data_std, ens_members


def get_ref_ds(dataset) -> xr.Dataset | None:
    """
    Get a reference reprojected ERA5 dataset from the specified source files.

    This function reads through the source JSON configuration files to locate
    and open the first valid NetCDF file that can be used as a reference.

    Args:
        dataset: A Dataset object containing a `loader_config` attribute
            pointing to the configuration file with source information.

    Returns:
        The reprojected/regridded reference ERA5 dataset loaded from NetCDF file.
    """
    with open(dataset.loader_config, "r") as fh:
        loader_config = orjson.loads(fh.read())

    # Find `processed_data/.../ua700_abs.nc` file to use as reference
    # Since I used rioxarray to set the spatial dims here when reprojecting
    # I can just base my output netCDF off of that file.
    for k_sources, v_sources in loader_config["sources"].items():
        if v_sources.get("source_files", None) is not None:
            for k_source_files, v_source_files in v_sources["source_files"].items():
                # Loop through what should be train, val, test splits
                # (doesn't matter which one I pick)
                for k_split, v_split in v_source_files.items():
                    if isinstance(v_split, list):
                        reference_nc_file = v_split[0]
                    else:
                        reference_nc_file = v_split
                    return xr.open_dataset(reference_nc_file)


def denormalise_ua700(
    loader_config_file: str, da: xr.DataArray, var_name: str = "ua700"
):
    """
    Denormalise a specific variable in an xarray DataArray using configuration
    from processed data files.

    This function reads through the source JSON configurations to locate and apply
    the appropriate denormalisation transformation for the specified variable.

    Args:
        loader_config_file: Path to the configuration file containing
            information about sources and their implementations.

        da: The xarray DataArray containing the data to be
            denormalized.

        var_name (optional): Name of the variable to denormalise.
            Defaults to "ua700".

    Returns:
        New denormalised xarray DataArray for the specified variable.

    Raises:
        KeyError: If the specified `var_name` is not found in any processed files.
    """
    # Get config file for the dataset under "processed/" (i.e. amalgamation of
    # different data sources from "processed_data/") right before generating
    # cached zarr datasets.
    loader_config = get_config(loader_config_file)

    for source in loader_config["sources"]:
        logging.info(
            f"{source} -> {loader_config['sources'][source]['implementation']}"
        )
        processed_config_file = loader_config["filenames"][source]
        processed_config = get_config(processed_config_file)["data"]

        # Point to symlinked directory for reference training dataset
        # used to created the prediction dataset, and to use here again
        # for denormalising the raw pytorch prediction output.
        ref_procdir = os.path.join(processed_config["base_path"], "ref_training_dataset")
        processed_config |= {"ref_procdir": ref_procdir}
        processed_implementation = get_processor_from_source(
            identifier=source, source_cfg=processed_config
        )
        if var_name in processed_implementation.abs_vars:
            logging.info(f"Denormalising xr.Dataarray using {var_name}")
            da_target_var_denormalised = (
                processed_implementation._normalise_array_scaling(
                    var_name, da, denormalise=True
                )
            )
            return da_target_var_denormalised

    raise KeyError(f"`{var_name}` variable not found in processed files")


def create_cf_output(cfg: DictConfig) -> None:
    """
    Create a CF-compliant NetCDF file from prediction outputs.

    This function processes prediction data for a given set of dates,
    constructs an xarray Dataset with appropriate metadata and coordinates,
    and saves it to a specified output directory in NetCDF format.

    Returns:
        The function saves the processed data as a NetCDF file.

    Notes:
        Based on `create_cf_output` class from the IceNet library.
            https://github.com/icenet-ai/icenet/blob/6caa234907904bfa76b8724d8c83cd989230494a/icenet/process/predict.py#L122
    """

    plain = cfg.postprocess.netcdf.plain
    seeds = cfg.predict.seed

    # Make seeds into a list of seed values
    if isinstance(seeds, int):
        seeds = [seeds]

    dates = [dt.date(*[int(v) for v in s.split("-")]) for s in cfg.predict.dates]

    predict_dir_root = cfg.paths.predict

    # Point to symlinked cache directory - use config file to get a reference netCDF
    # to parse metadata from
    cache_dir = os.path.join(predict_dir_root, CACHE_SYMLINK_DIR)

    # Get config file from cache dir
    dataset_config_file = glob.glob(os.path.join(cache_dir, "*.json"))[0]

    # TODO: This creates empty "./network_datasets" directory.
    ds = CANARIMLDataSetTorch(dataset_config_file)
    dl = ds.get_data_loader()
    hemi_str = "north" if dl.north else "south"

    # Use reference regridded/reprojected dataset with rioxarray projection details
    ds_ref = get_ref_ds(ds)

    # Get prediction data
    data_mean, data_std, data, ens_members = zip(
        *[
            get_prediction_data(
                predict_dir_root=predict_dir_root, seeds=seeds, date=date, return_ensemble_data=True
            )
            for date in dates
        ]
    )

    # (time, ensemble, xc, yc, leadtime)
    data = np.array(data)

    # (time, xc, yc, leadtime)
    data_mean = np.array(data_mean)
    data_std = np.array(data_std)

    logging.info("Dataset arr shape: {}".format(data.shape))

    # Get ensemble mean (denormalised) and std dev
    dataset_config = get_config(dataset_config_file)
    loader_config_file = Path(dataset_config["loader_config"])
    ua700_mean = denormalise_ua700(
        loader_config_file, data_mean, var_name="ua700"
    )
    ua700_stddev = data_mean

    ua700 = da.zeros_like(data)

    for i, ensemble_member in enumerate(seeds):
        ua700[:, i, ...] = denormalise_ua700(
            loader_config_file, ua700[:, i, ...], var_name="ua700"
        )

    data_vars = dict(
        ua700=(["time", "ensemble", "y", "x", "leadtime"], ua700),
        ua700_mean=(["time", "y", "x", "leadtime"], ua700_mean),
        ua700_stddev=(["time", "y", "x", "leadtime"], ua700_stddev),
        ensemble_members=(["ensemble"], seeds),
    )

    if hasattr(ds_ref, "spatial_ref"):
        data_vars |= dict(spatial_ref=ds_ref.spatial_ref)

    coords = dict(
        time=[pd.Timestamp(d) for d in dates],
        leadtime=np.arange(1, data_mean.shape[3] + 1, 1),
        ensemble=np.arange(len(seeds)),
    )

    extra_attrs = dict()

    ##
    # Metadata
    #
    if not plain:
        canari_ml_version = version(canari_ml.__name__)

        lists_of_fcast_dates = [
            [
                pd.Timestamp(
                    date
                    + relativedelta(
                        **{
                            f"{cfg.frequency.lower()}s": int(
                                lead_idx
                            )
                        } # type: ignore
                    )
                )
                for lead_idx in np.arange(1, data_mean.shape[3] + 1, 1)
            ]
            for date in dates
        ]

        # Assigning to parameters for dataarray
        coords["x"] = ds_ref.coords["x"].data
        coords["y"] = ds_ref.coords["y"].data
        coords["forecast_date"] = (("time", "leadtime"), lists_of_fcast_dates)

        extra_attrs = dict(
            canari_ml_mask_implementation="canari_ml.data.masks.era5:Masks",
            # spatial_resolution=ref_cube.attributes["spatial_resolution"],
            # Use ISO 8601:2004 duration format, preferably the extended format
            # as recommended in the Attribute Content Guidance section.
            time_coverage_start=min(
                set([item for row in lists_of_fcast_dates for item in row])
            ).isoformat(),
            time_coverage_end=max(
                set([item for row in lists_of_fcast_dates for item in row])
            ).isoformat(),
        )

        xarr = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            # REF: https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3
            attrs={
                **dict(
                    Conventions="CF-1.6 ACDD-1.3",
                    comments="",
                    creator_email="bryald@bas.ac.uk",
                    creator_institution="British Antarctic Survey",
                    creator_name="Bryn Noel Ubald",
                    creator_url="www.bas.ac.uk",
                    date_created=dt.datetime.now().strftime("%Y-%m-%d"),
                    geospatial_bounds_crs="EPSG:6931" if dl.north else "EPSG:6932",
                    # geospatial_vertical_min=0.0,
                    # geospatial_vertical_max=0.0,
                    hemisphere_string=hemi_str,
                    history="{} - creation".format(dt.datetime.now()),
                    id="Canari-ML {}".format(canari_ml_version),
                    institution="British Antarctic Survey",
                    # REF: https://gcmd.earthdata.nasa.gov/KeywordViewer/scheme/Earth Science/592d49c4-e8ae-4ab4-bf24-ae4a896d0637?gtm_keyword=UPPER LEVEL WINDS&gtm_scheme=Earth Science
                    keywords="""f'Earth Science > Atmosphere > Atmospheric Winds > Upper Level Winds > U WIND COMPONENT""",
                    keywords_vocabulary="GCMD Science Keywords",
                    license="Open Government Licece (OGL) V3",
                    naming_authority="uk.ac.bas",
                    platform="BAS HPC",
                    product_version=canari_ml_version,
                    project="Canari-ML",
                    publisher_email="",
                    publisher_institution="British Antarctic Survey",
                    # publisher_name="Polar Data Center",
                    publisher_url="",
                    source=f"""
                Canari-ML model generation at v{canari_ml_version}
                """,
                    # Values for any standard_name attribute must come from the CF
                    # Standard Names vocabulary for the data file or product to
                    #  comply with CF
                    standard_name_vocabulary="CF Standard Name Table v27",
                    summary="""
                This is an output of northern ua700 predictions from the
                Canari-ML model run in an ensemble, with postprocessing to determine
                the mean and standard deviation across the runs.
                """,
                    # TODO: Need to update this to pick up if daily or monthly run
                    # REF: https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3
                    # REF: https://www.digi.com/resources/documentation/digidocs/90001488-13/reference/r_iso_8601_duration_format.htm
                    # time_coverage_duration="P1D",
                    # time_coverage_resolution="P1D",
                    title="North Atlantic Zonal Wind Prediction",
                ),
                **extra_attrs,
            },
        )

    ##
    # Variable attributes
    #
    if not plain:
        xarr.time.attrs = dict(
            long_name="forecast initialisation time",
            standard_name="forecast_reference_time",
            axis="T",
            # TODO: https://github.com/SciTools/cf-units for units methods
            # units=Unit(
            #     "days since 1900-01-01 00:00:00", calendar="gregorian"
            # ).definition,
            # bounds=array([[31622400., 31708800.]])
        )

        xarr.y.attrs = dict(
            long_name="y coordinate of projection (northings)",
            standard_name="projection_y_coordinate",
            units=Unit("meters").definition,
            axis="Y",
        )

        xarr.x.attrs = dict(
            long_name="x coordinate of projection (eastings)",
            standard_name="x coordinate of projection (eastings)",
            units=Unit("meters").definition,
            axis="X",
        )

        xarr.leadtime.attrs = dict(
            long_name="leadtime of forecast in relation to reference time",
            short_name="leadtime",
            # TODO: days, months etc from ground_truth_ds_config.frequency.attribute
            # units="1",
        )

        xarr.ua700_mean.attrs = dict(
            long_name="Mean U component of wind at 700hPa",
            standard_name="eastward_wind",
            short_name="ua700",
            ancillary_variables="ua700_stddev",
            # grid_mapping="Lambert_Azimuthal_Grid",
            units=Unit("m/s").definition,
        )

        xarr.ua700_stddev.attrs = dict(
            long_name="total uncertainty (one standard deviation) of zonal wind at 700hPa",
            standard_name="eastward_wind standard_error",
            # grid_mapping="Lambert_Azimuthal_Grid",
            units="1",
        )

        xarr.ensemble_members.attrs = dict(
            long_name="seeds for ensemble members used to create this output",
            short_name="ensemble_members",
            units="1",
        )

        # TODO: split into daily files
        nc_path = cfg.paths.postprocess.netcdf_path
        nc_file = os.path.join(nc_path, cfg.postprocess.netcdf.name)
        if not Path(nc_path).exists():
            os.makedirs(nc_path, exist_ok=True)
        logging.info("Saving to {}".format(nc_file))
        xarr.to_netcdf(nc_file)
