import logging
import os
from pathlib import Path

import cartopy.crs as ccrs
import rioxarray
import xarray as xr
from download_toolbox.interface import DatasetConfig
from preprocess_toolbox.dataset.cli import init_dataset

from .cli import RegridArgParser

# Get the logger for rasterio (or the relevant library)
logger = logging.getLogger("rasterio")

# Set the logging level to WARNING to suppress INFO-level messages
logger.setLevel(logging.WARNING)


def regrid_dataset(netcdf_file, coarsen: int = 1, interpolate: bool = False):
    """
    Regrids an ERA5 dataset from its lat/lon grid to the LAEA using
    rioxarray.
    """
    ds = xr.open_dataset(netcdf_file, decode_coords="all")

    if not hasattr(ds, "spatial_ref"):
        logging.debug(
            "No spatial reference found in dataset, assuming geodetic (PlateCarree) grid"
        )
        # This will add a `.spatial_ref`` attribute to the dataset,
        # accessible via `ds.spatial_ref`.
        # Assume that dataset is a lat/lon grid
        ds.rio.write_crs(4326, inplace=True)

    laea = ccrs.LambertAzimuthalEqualArea(
        central_longitude=0, central_latitude=90
    ).to_string()
    ds_laea = ds.rio.reproject(laea)

    if interpolate:
        # Interpolate missing regions (below equator) for CANARI, might be
        # useful if training mask doesn't align exactly?
        ds_laea = ds_laea.rio.interpolate_na("nearest")

    if coarsen > 1:
        ds_laea = ds_laea.coarsen(x=2, y=2, boundary="exact").mean()

    return ds_laea


def regrid_datasets_from_config(
    process_config: DatasetConfig, coarsen: int = 1, interpolate: bool = False
):
    logging.info("Regridding dataset")

    for datafile in [
        _ for var_files in process_config.var_files.values() for _ in var_files
    ]:
        (datafile_path, datafile_name) = os.path.split(datafile)
        regrid_source_name = f"_regrid_{datafile_name}"
        regrid_datafile = Path(datafile_path) / regrid_source_name
        os.rename(datafile, regrid_datafile)

        logging.debug("Regridding {}".format(regrid_datafile))

        ds_laea = regrid_dataset(regrid_datafile, coarsen)

        logging.debug("Saving regridded data to {}... ".format(datafile))
        ds_laea.to_netcdf(datafile)

        if os.path.exists(datafile):
            os.remove(regrid_datafile)


def regrid():
    args = (
        RegridArgParser()
        .add_destination()
        .add_splits()
        .add_coarsen()
        .add_interpolate()
        .parse_args()
    )
    # Initially copy across the source data from `./data/` to the destination
    # `./processed_data/`
    ds, ds_config = init_dataset(args)
    regrid_datasets_from_config(
        ds_config, coarsen=args.coarsen, interpolate=args.interpolate
    )
    ds_config.save_config()
