"""Module to mask out the northern/southern hemisphere"""

import argparse
import logging
import os

import cartopy.crs as ccrs
import numpy as np
import pyproj
import xarray as xr
from download_toolbox.interface import (
    Configuration,
    DatasetConfig,
    get_dataset_config_implementation,
    get_implementation,
)
from preprocess_toolbox.cli import BaseArgParser
from preprocess_toolbox.loader.cli import MetaArgParser
from preprocess_toolbox.processor import Processor
from preprocess_toolbox.utils import get_config_filename, update_config
from scipy.ndimage import gaussian_filter


class MaskDatasetConfig(DatasetConfig):
    """Configuration class for generating ERA5 mask datasets.

    Inherits from download_toolbox.interface.DatasetConfig and extends it
    to handle hemisphere-specific masks.

    Attributes:
        variable_name: Name of the variable to process.
            Defaults to None.
        reference_era5_file: Path to reference ERA5 file for mask generation.
            Defaults to None.
    Notes:
        Based on `MaskDatasetConfig` class from the IceNet library.
            https://github.com/icenet-ai/icenet/blob/6caa234907904bfa76b8724d8c83cd989230494a/icenet/data/masks/osisaf.py
    """
    def __init__(
        self,
        downloaded_files: list = None,
        identifier: str = "masks",
        variable_name=None,
        reference_era5_file=None,
        base_weight=1.0,
        region_weights=None,
        weight_smoothing_sigma=10,
        **kwargs,
    ):
        """Initialise the MaskDatasetConfig class.

        Args:
            downloaded_files (optional): List of downloaded files.
                Defaults to None.
            identifier (optional): Identifier for this dataset configuration.
                Defaults to "masks".
            variable_name: Name of the ERA5 variable to process. Must be specified.
                Defaults to None.
            reference_era5_file: Path to the reference ERA5 file. Must be specified.
                Defaults to None.
            **kwargs: Additional keyword arguments passed to super class.

        Raises:
            ValueError: If either variable_name or reference_era5_file are None.
            NotImplementedError: If location is neither north nor south.
        """
        super().__init__(
            identifier=identifier,
            levels=[None, None],
            path_components=[],
            var_names=["hemisphere", "weighted_regions"],
            **kwargs,
        )

        if not self.location.north and not self.location.south:
            raise NotImplementedError("Location must be north or south, not custom")

        self._hemi_str = "nh" if self.location.north else "sh"

        if variable_name is None or reference_era5_file is None:
            raise ValueError(
                "Reference ERA5 variable name and corresponding reference file "
                "must be specified"
            )
        self.variable_name = variable_name
        self.reference_era5_file = reference_era5_file
        self._base_weight = base_weight
        self._region_weights = region_weights
        self._weight_smoothing_sigma = weight_smoothing_sigma

    def _load_reference_dataset(self) -> xr.Dataset:
        """Load the reference ERA5 dataset for mask generation.

        Returns:
            The loaded ERA5 dataset.
        """
        return xr.open_dataset(self.reference_era5_file, decode_coords="all")

    def _generate_hemisphere(self):
        """Generate a hemisphere mask based on latitude and longitude bounds.

        The mask is generated using the Lambert Azimuthal Equal Area projection
        and saved as a numpy array. The mask covers either northern or southern
        hemisphere based on the `self._hemi_str` attribute.
        """
        hemisphere_mask_path = os.path.join(
            self.path, "hemisphere_mask.{}.npy".format(self._hemi_str)
        )

        if not os.path.exists(hemisphere_mask_path):
            ds_ref = self._load_reference_dataset()

            try:
                # Extract the spatial reference that I had rioxarray write when
                # reprojecting (if it exists)
                crs_wkt = ds_ref.spatial_ref.attrs.get("crs_wkt", None)

                # If the spatial_ref is in EPSG code, use pyproj to define the CRS
                if crs_wkt:
                    # A WKT string
                    proj_crs = pyproj.CRS.from_wkt(crs_wkt)
                else:
                    raise ValueError("Unsupported spatial reference format")
            except AttributeError:
                logging.warning("Failed to read spatial reference from dataset"
                "Not been run through `rioxarray`?")
                # Define coordinate reference systems
                central_latitude = 90 if self._hemi_str == "nh" else -90
                proj_crs = ccrs.LambertAzimuthalEqualArea(
                    central_longitude=0, central_latitude=central_latitude
                )

            geodetic = ccrs.Geodetic()  # Geographic projection (lat/lon)

            # Define geographic region bounds to mask (latitude, longitude)
            if self._hemi_str == "nh":
                lat_min, lat_max = 0, 90
            else:
                lat_min, lat_max = -90, 0
            lon_min, lon_max = -180, 180

            X, Y = np.meshgrid(ds_ref.x.values, ds_ref.y.values)

            # Transform dataset's projected coordinates (x, y) to latitude and longitude
            lonlat = geodetic.transform_points(proj_crs, X, Y)
            lon = lonlat[..., 0]
            lat = lonlat[..., 1]

            mask = (
                (lat >= lat_min)
                & (lat <= lat_max)
                & (lon >= lon_min)
                & (lon <= lon_max)
            )

            # mask_xr = xr.DataArray(mask, coords={"y": ds.y, "x": ds.x}, dims=["y", "x"])

            logging.info(f"Saving {hemisphere_mask_path}")

            # Store flipped mask (i.e., want 1's across region we don't want)
            np.save(hemisphere_mask_path, ~mask)
        return hemisphere_mask_path

    def _generate_weighted_regions(self):
        """Generate a region based weighting by latitude and longitude bounds.

        The mask is generated using the Lambert Azimuthal Equal Area projection
        and saved as a numpy array.
        """
        base_weight = self._base_weight
        region_weights = self._region_weights
        weighted_regions_path = os.path.join(
            self.path, "weighted_regions.{}.npy".format(self._hemi_str)
        )

        ds_ref = self._load_reference_dataset()

        try:
            # Extract the spatial reference that I had rioxarray write when
            # reprojecting (if it exists)
            crs_wkt = ds_ref.spatial_ref.attrs.get("crs_wkt", None)

            # If the spatial_ref is in EPSG code, use pyproj to define the CRS
            if crs_wkt:
                # A WKT string
                proj_crs = pyproj.CRS.from_wkt(crs_wkt)
            else:
                raise ValueError("Unsupported spatial reference format")
        except AttributeError:
            logging.warning("Failed to read spatial reference from dataset"
            "Not been run through `rioxarray`?")
            # Define coordinate reference systems
            central_latitude = 90 if self._hemi_str == "nh" else -90
            proj_crs = ccrs.LambertAzimuthalEqualArea(
                central_longitude=0, central_latitude=central_latitude
            )

        geodetic = ccrs.Geodetic()  # Geographic projection (lat/lon)

        X, Y = np.meshgrid(ds_ref.x.values, ds_ref.y.values)

        # Transform dataset's projected coordinates (x, y) to latitude and longitude
        lonlat = geodetic.transform_points(proj_crs, X, Y)
        lon = lonlat[..., 0]
        lat = lonlat[..., 1]

        # Initialise sample weights to a base value everywhere
        # base_weight = 0.1
        sample_weights = np.full_like(lat, base_weight, dtype=np.float32)

        # Define region weights: (lat_min, lat_max, lon_min, lon_max, weight)
        # region_weights = [
        #     (20, 80, -100, 45, 0.25),
        #     (40, 60, -75, 15, 0.65),
        # ]

        # Controls Gaussian blur extent (in grid points)
        smoothing_sigma = self._weight_smoothing_sigma

        if region_weights:
            for lat_min, lat_max, lon_min, lon_max, weight in region_weights:
                # Hard mask of region
                region_mask = (
                    (lat >= lat_min)
                    & (lat <= lat_max)
                    & (lon >= lon_min)
                    & (lon <= lon_max)
                ).astype(np.float32)

                # Apply Gaussian smoothing
                smooth_mask = gaussian_filter(region_mask, sigma=smoothing_sigma)

                # Blend smoothly with existing weights — higher priority for later definitions
                sample_weights = (1 - smooth_mask) * sample_weights + smooth_mask * weight

        logging.info(f"Saving {weighted_regions_path}")
        np.save(weighted_regions_path, sample_weights)
        return weighted_regions_path

    def save_data_for_config(
        self,
        rename_var_list: dict = None,
        source_ds: object = None,
        source_files: list = None,
        time_dim_values: list = None,
        var_filter_list: list = None,
        **kwargs,
    ) -> None:
        """Save data for the current configuration.

        Processes each variable configuration and generates corresponding files.

        Args:
            rename_var_list: Dictionary mapping old to new variable names.
                Defaults to None.
            source_ds: Source dataset.
                Defaults to None.
            source_files: List of source files.
                Defaults to None.
            time_dim_values: Time dimension values.
                Defaults to None.
            var_filter_list: List of variables to filter.
                Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        for var_config in self.variables:
            files = getattr(self, "_generate_{}".format(var_config.name))()
            self.var_files[var_config.name] = files

    def get_config(self, config_funcs: dict = None, strip_keys: list = None) -> dict:
        """Get the configuration object with specified keys removed.

        Args:
            config_funcs: Dictionary of configuration functions.
                Defaults to None.
            strip_keys: List of keys to remove from the configuration.
                Defaults to None.

        Returns:
            The modified configuration object.
        """
        return super().get_config(
            strip_keys=[
                # "_filename_template_osi450",
                # "_hemi_str",
                "_identifier",
                "_levels",
                "_path_components",
                # "_retrieve_cmd_template_osi450",
                "_var_names",
                # "_year",
            ]
        )

    @property
    def config(self) -> dict:
        """Get the configuration object.

        If not already created, initialises a Configuration object with the location name.

        Returns:
            The dataset configuration object.
        """
        if self._config is None:
            logging.debug(
                "Creating dataset configuration with {}".format(self.location.name)
            )
            self._config = Configuration(
                config_type=self.config_type,
                directory=self.root_path,
                identifier=self.location.name,
            )
        return self._config


class Masks(Processor):
    """A Processor class for generating and applying hemisphere-specific masks.

    Inherits from `preprocess_toolbox.processor.Processor` to handle mask
    generation and data processing, particularly for ERA5 datasets.
    This class manages the creation of northern or southern hemisphere
    masks based on configuration settings.

    Attributes:
        _dataset_config (DatasetConfig): Configuration object containing dataset
            parameters, including location, variables, and file paths.
        abs_vars (list): List of variables treated as absolute in processing.
        _hemi_str (str): 'north' or 'south', indicating which hemisphere is being
            processed.
        _region (tuple): Slice/slices defining the region to apply masking.

    Notes:
        Based on `Masks` class from the IceNet library.
            https://github.com/icenet-ai/icenet/blob/6caa234907904bfa76b8724d8c83cd989230494a/icenet/data/masks/osisaf.py
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        *args,
        absolute_vars: list = None,
        identifier: str = None,
        base_weight: float = 1.0,
        region_weights: tuple | list = None,
        weight_smoothing_sigma: float = 10.0,
        **kwargs,
    ):
        """Initialise the Masks processor with configuration settings.

        Args:
            dataset_config: Configuration object for the dataset.
            *args: Additional positional arguments passed to super class.
            absolute_vars (optional): Variables treated as absolute.
                Defaults to None.
            identifier (optional): Identifier for processing.
                Defaults to None.
            **kwargs: Additional keyword arguments passed to super class.
        """

        # Use first ERA5 variable available to get a netcdf reference
        variable_name = next(iter(dataset_config.variables)).name

        # Use first ERA5 file available from this variable as reference file
        reference_era5_file = dataset_config.var_files.get(variable_name, None)[0]

        mask_ds = MaskDatasetConfig(
            base_path=dataset_config.base_path,
            frequency=dataset_config.frequency,
            location=dataset_config.location,
            variable_name=variable_name,
            reference_era5_file=reference_era5_file,
            base_weight=base_weight,
            region_weights=region_weights,
            weight_smoothing_sigma=weight_smoothing_sigma,
        )
        mask_ds.save_data_for_config()
        self._dataset_config = mask_ds.save_config()
        self._hemi_str = "north" if dataset_config.location.north else "south"

        super().__init__(
            mask_ds,
            absolute_vars=["hemisphere", "weighted_regions"],
            # dtype=np.dtype(bool), # Removed since "weighted_regions" is a float,
                                    # dealt with in `generate_sample` func.
            identifier="masks.{}".format(self._hemi_str),
            **kwargs,
        )

        self._source_files = mask_ds.var_files.copy()
        self._region = (slice(None, None), slice(None, None))

    def get_config(self, config_funcs: dict = None, strip_keys: list = None) -> dict:
        """Retrieve the configuration dictionary for the processor.

        Args:
            config_funcs (optional): Dictionary of functions to modify config.
                Defaults to None.
            strip_keys (optional): Keys to remove from the config.
                Defaults to None.

        Returns:
            dict: Configuration dictionary containing module and class implementation,
                absolute variables, dataset configuration, path, processed files,
                and source files.
        """
        return {
            "implementation": "{}:{}".format(self.__module__, self.__class__.__name__),
            "absolute_vars": self.abs_vars,
            "dataset_config": self._dataset_config,
            "path": self.path,
            "processed_files": self._processed_files,
            "source_files": self._source_files,
        }

    def process(self):
        """Generate and save the hemisphere mask based on the configured region."""
        # Hemisphere mask preparation
        hemisphere_mask = np.load(self._source_files["hemisphere"])

        da_hemisphere_mask = xr.DataArray(
            data=hemisphere_mask,
            dims=["y", "x"],
            attrs=dict(description="Mask of hemisphere"),
        )

        self.save_processed_file(
            "hemisphere",
            os.path.basename(self.hemisphere_filename),
            da_hemisphere_mask,
            overwrite=True,
        )

        # Hemisphere mask preparation
        weighted_regions = np.load(self._source_files["weighted_regions"])

        da_weighted_regions = xr.DataArray(
            data=weighted_regions,
            dims=["y", "x"],
            attrs=dict(description="Weighted regions"),
        )

        self.save_processed_file(
            "weighted_regions",
            os.path.basename(self.weighted_regions_filename),
            da_weighted_regions,
            overwrite=True,
        )

        self.save_config()

    def hemisphere(self, *args, **kwargs) -> xr.DataArray:
        """Return the hemisphere mask as an xr.DataArray.

        Args:
            *args:
            **kwargs:

        Returns:
            xr.DataArray: The hemisphere mask loaded from the specified file.
        """
        da = xr.open_dataarray(self.hemisphere_filename)
        return da.data[self._region]

    def weighted_regions(self, *args, **kwargs) -> xr.DataArray:
        """Return the weighted regions as an xr.DataArray.

        Args:
            *args:
            **kwargs:

        Returns:
            xr.DataArray: The hemisphere mask loaded from the specified file.
        """
        da = xr.open_dataarray(self.weighted_regions_filename)
        return da.data[self._region]

    def get_blank_mask(self) -> np.array:
        """Returns an empty boolean mask for the configured region.

        Returns:
            A boolean array of shape matching the hemisphere mask,
                initialised to `False` for the pre-defined `self._region`.
        """
        shape = self.hemisphere().shape
        return np.full(shape, False)[self._region]

    def __getitem__(self, item):
        """Sets slice of region wanted for masking, and allows method chaining.

        Args:
            item: Index/slice to extract and set as the new region.

        Returns:
            self: For method chaining.
        """
        logging.info("Mask region set to: {}".format(item))
        self._region = item
        return self

    @property
    def region(self) -> tuple:
        """Get the current mask region.

        Returns:
            The current region slices used for masking.
        """
        return self._region

    @region.setter
    def region(self, value: tuple) -> None:
        """Set a new region for masking.

        Args:
            value: New region slices to apply.
        """
        self._region = value

    def reset_region(self):
        """Resets the mask region to cover the entire dataset."""
        logging.info("Mask region reset, whole mask will be returned")
        self._region = (slice(None, None), slice(None, None))

    @property
    def hemisphere_filename(self) -> str:
        """Get the filename for the hemisphere mask.

        Returns:
            Path to the hemisphere mask file.
        """
        return os.path.join(self.path, "hemisphere.{}.nc".format(self._hemi_str))

    @property
    def weighted_regions_filename(self) -> str:
        """Get the filename for the weighted regions.

        Returns:
            Path to the weighted regions file.
        """
        return os.path.join(self.path, "weighted_regions.{}.nc".format(self._hemi_str))


class RegionWeightAction(argparse.Action):
    """
    Custom argparse action for handling region weights with their respective boundaries.

    This action expects 5 values: `lat_min`, `lat_max`, `lon_min`, `lon_max`, and
    `weight`.
    It supports passing these values as a comma-separated or space-separated string.
    If the number of provided values is not 5, an error is raised.
    All values must be numeric.

    This action accumulates region weights in the `namespace` object under the
    `region_weights` attribute, allowing multiple regions to be specified by calling
    the flag repeatedly.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        # Case of comma separated string
        if (
            len(values) == 1
            and isinstance(values[0], str)
            and ("," in values[0] or " " in values[0])
        ):
            delimiter = "," if "," in values[0] else " " if " " in values[0] else None
            if delimiter:
                values = [v.strip() for v in values[0].split(delimiter) if len(v)]

        if len(values) != 5:
            raise argparse.ArgumentError(
                self,
                "Must provide 5 values: lat_min, lat_max, lon_min, lon_max, weight",
            )

        try:
            lat_min, lat_max, lon_min, lon_max, weight = map(float, values)
        except ValueError:
            raise argparse.ArgumentError(self, "All values must be numeric.")

        # Account for the fact that region_weights can be passed multiple times
        if not hasattr(namespace, "region_weights"):
            setattr(namespace, "region_weights", [])

        region = (lat_min, lat_max, lon_min, lon_max, weight)
        namespace.region_weights.append(region)


class WeightsArgParser(MetaArgParser):
    """
    Argument parser for handling region weights.

    This class extends :class:`MetaArgParser` and adds arguments related to managing
    region weights. It supports specifying a base weight, individual region weights,
    and smoothing of the weights using a Gaussian kernel with a given sigma.
    """

    def __init__(self):
        super().__init__()

    def add_region_weights(self):
        self.add_argument(
            "--base-weight",
            type=float,
            required=True,
            default=1.0,
            help="Base weight (float)",
        )
        self.add_argument(
            "--region-weights",
            nargs="+",
            action=RegionWeightAction,
            metavar="REGION",
            help="Specify a region and its weight. Can be used multiple times, but, weights must sum to 1.0.",
            default=[],
        )
        self.add_argument(
            "--weight-smoothing-sigma",
            type=float,
            default=10,
            help="Sigma for Gaussian smoothing of region weights (float)",
        )
        return self


def get_channel_info_from_processor(cfg_segment: str):
    """
    Retrieves channel-specific information from a processor based on the given configuration segment.

    This function uses :class:`WeightsArgParser` to parse arguments related to channels and region weights.
    It then retrieves the appropriate implementation for the processor, dataset configuration, and initializes
    the processor with the parsed arguments. The processor is used to process data and obtain channel-specific
    information, which is then stored in a configuration file under the specified segment.

    Note:
        Based on code from `preprocess-toolbox`:
        https://github.com/environmental-forecasting/preprocess-toolbox/blob/35f57eecd8017fae0bf1c7a4a4ca80ca77e905d4/preprocess_toolbox/loader/cli.py#L131-L151

    Args:
        cfg_segment (str): The configuration segment under which to store the channel-specific information.

    Raises:
        RuntimeError: If the `--config-path` argument is provided, as it is invalid for this CLI endpoint.
    """
    args = WeightsArgParser().add_channel().add_region_weights().parse_args()

    proc_impl = get_implementation(args.implementation)
    ds_config = get_dataset_config_implementation(args.ground_truth_dataset)

    if args.config is not None:
        # FIXME: args.config contains the location of the dataset config on render, but
        #   this is not part of this pattern! DS is either ground truth or in derived class,
        #   but this library doesn't care or know of it respectively.
        raise RuntimeError("--config-path is invalid for this CLI endpoint, sorry...")

    # Overwrite argparse default for this func if base path not provided
    base_path = args.destination_path
    if base_path == "processed_data":
        base_path = os.path.join(".", "processed")

    total_weight = args.base_weight + sum(
        w[-1] for w in getattr(args, "region_weights", [])
    )
    if not abs(total_weight - 1.0) < 1e-6:
        logging.error(f"Total weight must sum to 1.0, not {total_weight:.3f}")

    processor = proc_impl(
        ds_config,
        [
            args.channel_name,
        ],
        args.channel_name,
        base_path=base_path,
        base_weight=args.base_weight,
        region_weights=args.region_weights,
        weight_smoothing_sigma=args.weight_smoothing_sigma,
    )
    processor.process()
    update_config(
        get_config_filename(args),
        cfg_segment,
        {args.channel_name: processor.get_config()},
    )


def add_region_weights():
    """Entry point for region weighted processing."""
    get_channel_info_from_processor("masks")
