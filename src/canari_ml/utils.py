import argparse
import datetime as dt
import re


def location_arg(argument: str) -> tuple[int, int]:
    """
    Parse a string argument as a location, expecting a pair of integers separated by a comma.

    Args:
        argument: The string argument containing the location.

    Returns:
        Tuple of the x and y coordinates of the location.

    Raises:
        argparse.ArgumentTypeError: If the argument cannot be parsed as a location.
    """
    try:
        x, y = parse_location_or_region(argument)
        return x, y
    except ValueError:
        argparse.ArgumentTypeError(
            "Expected a location (pair of integers separated by a comma)"
        )


def parse_location_or_region(argument: str, separator: str = ",") -> tuple[int, int]:
    """
    Parse a string argument as a sequence of integers separated by a specified separator.

    This function splits the input string using the given separator and attempts to convert
    each resulting substring into an integer.

    Args:
        argument: The string argument containing the sequence of integers.
        separator (optional): The character used to separate the integers in the string.
            Defaults to ",".

    Returns:
        Tuple of parsed integers.

    Raises:
        ValueError: If any substring cannot be converted into an integer.
    """
    return tuple(int(s) for s in argument.split(separator))


def region_arg(argument: str) -> tuple[int, int, int, int]:
    """
    Parse a string argument as a region, expecting four integers separated by commas.

    Args:
        argument: The string argument containing the region's coordinates.

    Returns:
        Tuple containing the x1, y1, x2, and y2 coordinates of the region.

    Raises:
        argparse.ArgumentTypeError: If the argument cannot be parsed as a valid region.
        RuntimeError: If the provided coordinates do not form a valid rectangle.
    """
    try:
        x1, y1, x2, y2 = parse_location_or_region(argument)

        if x2 < x1 or y2 < y1:
            raise RuntimeError(f"Region is not valid x1 {x1}:x2 {x2}, y1 {y1}:y2 {y2}")
        return x1, y1, x2, y2
    except TypeError:
        raise argparse.ArgumentTypeError(
            "Region argument must be list of four integers"
        )


def date_arg(string: str) -> dt.date:
    """
    Parse a string argument as a date in the format YYYY-MM-DD.

    Args:
        string: The string argument containing the date.

    Returns:
        The parsed date.

    Raises:
        argparse.ArgumentTypeError: If the argument cannot be parsed as a valid date.
    """
    date_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", string)
    return dt.date(*[int(s) for s in date_match.groups()])
