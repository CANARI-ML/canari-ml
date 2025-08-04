import argparse
import datetime as dt
import glob
import logging
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import xarray as xr
from cf_units import Unit
from dateutil.relativedelta import relativedelta
from download_toolbox.interface import get_dataset_config_implementation
from preprocess_toolbox.interface import get_processor_from_source
from preprocess_toolbox.utils import get_config

import canari_ml
from canari_ml.data.dataloader import CANARIMLDataSetTorch
from canari_ml.data.masks.era5 import Masks


def get_prediction_data(
    root: str, name: str, date: dt, return_ensemble_data: bool = False
) -> tuple:
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

    glob_str = os.path.join(
        root, "results", "predict", name, "*", date.strftime("%Y_%m_%d.npy")
    )

    np_files = glob.glob(glob_str)
    if not len(np_files):
        logging.warning("No files found")
        return None

    data = [np.load(f) for f in np_files]
    data = np.array(data)
    ens_members = data.shape[0]

    logging.debug("Data read from disk: {} from: {}".format(data.shape, np_files))

    data_mean = np.stack([data.mean(axis=0), data.std(axis=0)], axis=-1).squeeze()

    if return_ensemble_data:
        return data_mean, data, ens_members
    else:
        return data_mean, ens_members


def get_ref_ds(dataset) -> xr.Dataset:
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
        logging.debug(
            f"{source} -> {loader_config['sources'][source]['implementation']}"
        )
        processed_config_file = loader_config["filenames"][source]
        processed_config = get_config(processed_config_file)["data"]
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


def create_cf_output() -> None:
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
    args = get_args()

    with open(args.datefile_csv, "r") as f:
        dates = [dt.date(*[int(v) for v in s.split("-")]) for s in f.read().split()]

    dataset_config_file = os.path.join(args.root, f"dataset_config.{args.dataset}.json")

    ds = CANARIMLDataSetTorch(dataset_config_file)
    dl = ds.get_data_loader()
    hemi_str = "north" if dl.north else "south"

    # Use reference regridded/reprojected dataset with rioxarray projection details
    ds_ref = get_ref_ds(ds)

    arr, ens_members = zip(
        *[get_prediction_data(args.root, args.prediction_name, date) for date in dates]
    )
    ens_members = list(ens_members)
    arr = np.array(arr)

    logging.info("Dataset arr shape: {}".format(arr.shape))

    # Get ensemble mean (denormalised) and std dev
    dataset_config = get_config(dataset_config_file)
    loader_config_file = Path(dataset_config["loader_config"]).name
    ua700_mean = denormalise_ua700(loader_config_file, arr[..., 0], var_name="ua700")
    ua700_stddev = arr[..., 1]

    data_vars = dict(
        ua700_mean=(["time", "y", "x", "leadtime"], ua700_mean),
        ua700_stddev=(["time", "y", "x", "leadtime"], ua700_stddev),
        ensemble_members=(["time"], ens_members),
    )

    if hasattr(ds_ref, "spatial_ref"):
        data_vars |= dict(spatial_ref=ds_ref.spatial_ref)

    coords = dict(
        time=[pd.Timestamp(d) for d in dates],
        leadtime=np.arange(1, arr.shape[3] + 1, 1),
    )

    extra_attrs = dict()

    ##
    # Metadata
    #
    if not args.plain:
        canari_ml_version = version(canari_ml.__name__)

        ground_truth_ds_filename = "data.aws.day.north.json".format(
            hemi_str
        )
        ground_truth_ds_config = get_dataset_config_implementation(
            ground_truth_ds_filename
        )

        lists_of_fcast_dates = [
            [
                pd.Timestamp(
                    date
                    + relativedelta(
                        **{
                            f"{ground_truth_ds_config.frequency.attribute}s": int(
                                lead_idx
                            )
                        }
                    )
                )
                for lead_idx in np.arange(1, arr.shape[3] + 1, 1)
            ]
            for date in dates
        ]

        # Assigning to parameters for dataarray
        coords["x"] = ds_ref.coords["x"].data
        coords["y"] = ds_ref.coords["y"].data
        coords["forecast_date"] = (("time", "leadtime"), lists_of_fcast_dates)

        extra_attrs = dict(
            canari_ml_ground_truth_ds=ground_truth_ds_filename,
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
                    # geospatial_bounds_crs="EPSG:6931" if dl.north else "EPSG:6932",
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
    if not args.plain:
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
            grid_mapping="Lambert_Azimuthal_Grid",
            units=Unit("m/s").definition,
        )

        xarr.ua700_stddev.attrs = dict(
            long_name="total uncertainty (one standard deviation) of zonal wind at 700hPa",
            standard_name="eastward_wind standard_error",
            grid_mapping="Lambert_Azimuthal_Grid",
            units="1",
        )

        xarr.ensemble_members.attrs = dict(
            long_name="number of ensemble members used to create this prediction",
            short_name="ensemble_members",
            units="1",
        )

        # dataset_config = get_config(dataset_config_file)
        # loader_config_file = Path(dataset_config["loader_config"]).name
        # xarr["ua700_mean"] = denormalise_ua700(
        #     loader_config_file, xarr.ua700_mean, var_name="ua700"
        # )

        # TODO: split into daily files
        output_path = os.path.join(args.output_dir, f"{args.prediction_name}.nc")
        logging.info("Saving to {}".format(output_path))
        xarr.to_netcdf(output_path)


def get_args():
    """Get CLI arguments and parse them"""
    ap = argparse.ArgumentParser()
    ap.add_argument("prediction_name")
    ap.add_argument("dataset")
    ap.add_argument("datefile_csv")

    ap.add_argument(
        "-p",
        "--plain",
        default=False,
        help="Don't try to add geospatial or complex metadata from ground truth",
        action="store_true",
    )

    # TODO: Add option to also include ensemble member predictions in output
    ap.add_argument(
        "-e",
        "--ensemble",
        default=False,
        help="TODO: Also include ensemble member predictions in output",
        action="store_true",
    )

    ap.add_argument("-o", "--output-dir", default=".")
    ap.add_argument("-r", "--root", type=str, default=".")

    ap.add_argument("-v", "--verbose", action="store_true", default=False)

    return ap.parse_args()
