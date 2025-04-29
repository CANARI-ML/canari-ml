import logging
import os
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np
import rioxarray
import xarray as xr
from affine import Affine
from download_toolbox.interface import DatasetConfig
from joblib import Parallel, delayed
from preprocess_toolbox.dataset.cli import init_dataset
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from .cli import ReprojectArgParser, parse_crs, parse_shape

# Get the logger for rasterio (or the relevant library)
logger = logging.getLogger("rasterio")

# Set the logging level to WARNING to suppress INFO-level messages
logger.setLevel(logging.WARNING)


def reproject_dataset(
    netcdf_file: str,
    source_crs: str | None = None,
    target_crs: str | None = None,
    resolution: float | tuple[float, float] | None = None,
    shape: str | int | tuple[int, int] | None = None,
    target_transform: Affine | None = None,
    coarsen: int = 1,
    interpolate_nans: bool = False,
):
    """
    Reprojects a source dataset from source_crs to target_crs using
    rioxarray.
    """
    if isinstance(netcdf_file, xr.Dataset) or isinstance(netcdf_file, xr.DataArray):
        ds = netcdf_file
    else:
        ds = xr.open_dataset(netcdf_file, decode_coords="all")

    source_crs = source_crs if source_crs else "EPSG:4326"
    target_crs = target_crs if target_crs else "EPSG:6931"

    if not hasattr(ds, "spatial_ref"):
        logging.debug(
            f"No spatial reference found in dataset, setting grid to: {source_crs}"
        )
        # This will add a `.spatial_ref`` attribute to the dataset,
        # accessible via `ds.spatial_ref`.
        # Assume that dataset is a lat/lon grid if `source_crs` is not defined
        ds.rio.write_crs(source_crs, inplace=True)

    if not isinstance(shape, tuple):
        shape: tuple[int, int] = parse_shape(shape)

    ds_reprojected = ds.rio.reproject(
        target_crs,
        resolution=resolution,
        shape=shape,
        # TODO: Missing antimeridian slice issue with Polar reprojection when using
        # other resampling methods (e.g., bilinear, cubic)
        resampling=Resampling.nearest,
        nodata=np.nan,
        transform=target_transform,
    )

    if interpolate_nans:
        # Interpolate missing regions (nans), for CANARI, its below equator, might be
        # useful if training mask doesn't align exactly?
        ds_reprojected = ds_reprojected.rio.interpolate_na("nearest")

    if coarsen > 1:
        ds_reprojected = ds_reprojected.coarsen(
            x=coarsen, y=coarsen, boundary="trim"
        ).mean()

    # TODO: Storing grid mapping attributes in the dataset to be CF Compliant
    # This is more trouble than its worth currently - need to map the different
    # projections which expect different attributes.
    # target_crs_dict = target_crs.to_dict()
    # semi_major_axis=target_crs.ellipsoid.semi_major_metre
    # semi_minor_axis=target_crs.ellipsoid.semi_minor_metre
    # latitude_of_projection_origin=target_crs_dict.get('lat_0', 0)
    # longitude_of_projection_origin=target_crs_dict.get('lon_0', 0)
    # false_easting=target_crs_dict.get('x_0', 0)
    # false_northing=target_crs_dict.get('y_0', 0)

    # grid_mapping_attrs = {
    #     "grid_mapping_name": "lambert_azimuthal_equal_area",
    #     "longitude_of_projection_origin": longitude_of_projection_origin,
    #     "latitude_of_projection_origin": latitude_of_projection_origin,
    #     "false_easting": false_easting,
    #     "false_northing": false_northing,
    #     "semi_major_axis": semi_major_axis,
    #     "semi_minor_axis": semi_minor_axis,
    # }

    # # Add a new variable for the grid_mapping to the dataset
    # ds_reprojected["projection"] = xr.DataArray(np.zeros(()), attrs=grid_mapping_attrs)

    # # Link the 'data' variable to the 'projection' variable (through grid_mapping attribute)
    # ds_reprojected["tas"].attrs["grid_mapping"] = "projection"

    return ds_reprojected


def reproject_dataset_ease2(
    *args,
    **kwargs,
):
    """Reproject a dataset to EASE-Grid 2.0 standard"""

    target_crs = kwargs["target_crs"]
    if target_crs is None:
        raise ValueError("target_crs must be specified")
    shape = kwargs.get("shape", (720, 720))
    kwargs["shape"] = shape

    if kwargs.get("resolution", None):
        raise ValueError(
            f"Resolution cannot be specified for EASE-Grid 2.0: {kwargs['resolution']}"
        )

    if not isinstance(shape, tuple):
        shape: tuple[int, int] = parse_shape(shape)

    if target_crs == "EPSG:6931" or target_crs == "EPSG:6932":
        # Define grid parameters for EASE-Grid 2.0 standard grid
        # Reference: https://nsidc.org/data/user-resources/help-center/guide-ease-grids#anchor-25-km-resolution-ease-grids
        # `cell_size` is the grid resolution in meters taken from the table in above link
        if shape == (720, 720):
            cell_size = 25000
        elif shape == (500, 500):
            cell_size = 36000
        else:
            raise ValueError(
                f"shape doesn't match expected EASE-Grid 2.0 standard grid:\n\t(`{shape}`)"
            )
        # The x-axis coordinate of the outer edge of the upper-left pixel
        x0 = -9000000.0
        # The y-axis coordinate of the outer edge of the upper-left pixel
        y0 = 9000000.0
    else:
        raise ValueError(
            f"target_crs doesn't match expected EASE-Grid 2.0 standard grid:\n\t(`{target_crs}`)"
        )

    # Create an affine transform for the target grid.
    # from_origin expects (upper-left x, upper-left y, x resolution, y resolution)
    target_transform = from_origin(x0, y0, cell_size, cell_size)

    # # Define the target affine transform for EASE-Grid 2.0 (Northern Hemisphere)
    # # Here, the pixel size is 25,000 m, with the top-left corner at (-9000000, 9000000)
    # target_transform = Affine(25000, 0, -9000000,
    #                           0, -25000, 9000000)

    ds_reprojected = reproject_dataset(
        *args, target_transform=target_transform, **kwargs
    )

    return ds_reprojected


def reproject_datasets_from_config(
    process_config: DatasetConfig, ease2=False, workers: int=1, **kwargs
):
    logging.info("Reprojecting dataset")

    datafiles = [
        _ for var_files in process_config.var_files.values() for _ in var_files
    ]

    def reproject_file(datafile):
        try:
            (datafile_path, datafile_name) = os.path.split(datafile)
            reproject_source_name = f"_reproject_{datafile_name}"
            reproject_datafile = Path(datafile_path) / reproject_source_name
            os.rename(datafile, reproject_datafile)

            logging.debug(f"Reprojecting {reproject_datafile}")

            if ease2:
                ds_reprojected = reproject_dataset_ease2(
                    netcdf_file=reproject_datafile, **kwargs
                )
            else:
                ds_reprojected = reproject_dataset(netcdf_file=reproject_datafile, **kwargs)

            logging.debug(f"Saving reprojected data to {datafile}... ")
            ds_reprojected.to_netcdf(datafile)
        except Exception as e:
            print(f"Error reprojecting {datafile}: {e}")
            raise
        finally:
            # Ensure temp file is deleted
            if os.path.exists(reproject_datafile):
                os.remove(reproject_datafile)

    # Parallel(n_jobs=workers, backend="threading", verbose=13)(
    #     delayed(reproject_file)(datafile) for datafile in datafiles
    # )

    logging.info(f"{len(datafiles)} files to reproject")
    if workers > 1:
        logging.info(f"Reprojecting using {workers} workers")
        # _ = thread_map(reproject_file, datafiles, max_workers=workers)
        Parallel(n_jobs=workers, backend="loky", timeout=9999, verbose=51)(
            delayed(reproject_file)(datafile) for datafile in datafiles
        )
    else:
        logging.info("Reprojecting using one worker")
        for datafile in datafiles:
            reproject_file(datafile)

    logging.info("Reprojection completed")

def reproject():
    args = (
        ReprojectArgParser()
        .add_destination()
        .add_splits()
        .add_source_crs()
        .add_target_crs()
        .add_resolution()
        .add_shape()
        .add_ease2()
        .add_coarsen()
        .add_interpolate_nans()
        .parse_args()
    )
    # Initially copy across the source data from `./data/` to the destination
    # `./processed_data/`
    ds, ds_config = init_dataset(args)
    # Reproject and overwrite the copied data
    reproject_datasets_from_config(
        ds_config,
        source_crs=args.source_crs,
        target_crs=args.target_crs,
        resolution=args.resolution,
        shape=args.shape,
        ease2=args.ease2,
        coarsen=args.coarsen,
        interpolate_nans=args.interpolate_nans,
        workers=args.workers,
    )
    ds_config.save_config()
