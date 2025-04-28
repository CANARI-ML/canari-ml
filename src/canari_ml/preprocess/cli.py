import ast
import importlib
import logging

import cartopy.crs as ccrs
from preprocess_toolbox.dataset.cli import ProcessingArgParser


class ReprojectArgParser(ProcessingArgParser):
    def __init__(self):
        super().__init__()
        self.add_argument("-w", "--workers", default=1, type=int)

    def add_source_crs(self):
        self.add_argument(
            "--source-crs",
            type=str,
            required=False,
            default="ccrs.PlateCarree()",
            help="Source dataset CRS definition: Full cartopy.crs expression (e.g., ccrs.PlateCarree()",
        )
        return self

    def add_target_crs(self):
        self.add_argument(
            "--target-crs",
            type=str,
            required=False,
            default="ccrs.epsg(6931)",
            help="Source dataset CRS definition: Full cartopy.crs expression (e.g., ccrs.LambertAzimuthalEqualArea(central_longitude=0, central_latitude=90))",
        )
        return self

    def add_resolution(self):
        self.add_argument(
            "--resolution",
            type=float,
            required=False,
            default=None,
            help="Resolution of output grid (in meters or degrees). Can only specify either `--resolution` or `--shape`, not both",
        )
        return self

    def add_shape(self):
        self.add_argument(
            "--shape",
            type=str,
            required=False,
            default="720,720",
            help="Shape of output grid (in pixels, e.g. '720,720'). Can only specify either `--resolution` or `--shape`, not both",
        )
        return self

    def add_ease2(self):
        self.add_argument(
            "--ease2",
            action="store_true",
            help="Enable to output an EASE-Grid 2.0 conformal grid",
        )
        return self

    def add_coarsen(self):
        self.add_argument(
            "--coarsen",
            type=int,
            default=1,
            help="To coarsen output grid by this integer factor.",
        )
        return self

    def add_interpolate_nans(self):
        self.add_argument(
            "--interpolate-nans",
            action="store_true",
            help="Enable nearest neighbour interpolation to fill in missing areas.",
        )
        return self


def parse_shape(value: str) -> tuple[int, int]:
    """
    Parse a shape argument into a tuple of integers.

    This function takes a string representing shape dimensions.
    If the input is a single value, it is duplicated to form a tuple of
    length two. If multiple values are provided, they are converted into
    a tuple.

    Args:
        value: A string containing one or more integers separated by commas,
            representing shape dimensions.

    Returns:
        A tuple of two integers derived from the input string.

    Examples:
        parse_shape("5") returns (5, 5)
        parse_shape("5,6") returns (5, 6)
    """
    if isinstance(value, int):
        return (value, value)
    else:
        values = value.split(",")

        # If only one value is provided, repeat it to create a tuple of length 2
        if len(values) == 1:
            values.append(values[0])
        values = map(int, values)

    return tuple(values)


def parse_crs(crs_string: str):
    """Parse a string representing a Cartopy CRS expression into a CRS object.

    Args:
        crs_string: A string specifying the Cartopy CRS, such as 'ccrs.PlateCarree()' or
            'cartopy.crs.Mercator(central_longitude=0)'.

    Returns:
        A Cartopy CRS instance corresponding to the input string.

    Raises:
        ValueError: If the provided string does not conform to expected formats,
            or if it doesn't result in a valid CRS object.

    Examples:
        >>> parse_crs('ccrs.PlateCarree()')
        <cartopy.crs.crs.PlateCarree object at ...>

        >>> parse_crs('cartopy.crs.Mercator(central_longitude=0)')
        <cartopy.crs.mercator.Mercator object at ...>
    """
    # Extract the module and class name
    crs_expr = crs_string.strip()
    if not crs_expr.startswith("ccrs.") and not crs_expr.startswith("cartopy.crs."):
        raise ValueError(
            f"CRS expression must start with 'ccrs.' or 'cartopy.crs.':\n\t`{crs_expr}`"
        )

    # Get class name from the CRS expression (e.g., LambertAzimuthalEqualArea)
    class_name = crs_expr.strip("ccrs.").strip("cartopy.crs")
    class_name = crs_expr[5:].split("(")[0]

    # Capture the CRS parameters from the CRS expression
    # (e.g., central_longitude=0, central_latitude=90)
    class_args = crs_expr[5 + len(class_name) :]

    crs_module = importlib.import_module("cartopy.crs")

    crs_class = getattr(crs_module, class_name)

    # Clean up the arguments string and convert it into a valid Python expression
    # TODO: Revisit, using ast for safety...
    class_args = class_args.strip("(").strip(")")
    if class_args:
        class_args_dict = dict(arg.split("=") for arg in class_args.split(","))
        class_args_dict = {
            k.strip(): ast.literal_eval(v.strip()) for k, v in class_args_dict.items()
        }
    else:
        class_args_dict = {}
    crs = crs_class(**class_args_dict)

    if not isinstance(crs, ccrs.CRS):
        raise ValueError(
            f"The provided expression did not return a valid Cartopy CRS object: {crs}"
        )

    return crs
