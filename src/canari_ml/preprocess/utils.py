import ast
import importlib
import logging
import os
from hashlib import shake_256
from types import SimpleNamespace

import cartopy.crs as ccrs
import orjson
from omegaconf import Container, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


class IterableNamespace(SimpleNamespace):
    """Allows contains check"""
    def __contains__(self, key):
        """Enable `if x is in IterableNameSpaceObject`"""
        return hasattr(self, key)


def generate_hash(inputs: list) -> str:
    """Generate a SHAKE-256 hash from input data.

    Reference [this page](Ref: https://www.doc.ic.ac.uk/~nuric/posts/coding/how-to-hash-a-dictionary-in-python/)

    Args:
        inputs: List of input data to be hashed. The contents are serialized
            using orjson with sorted keys for deterministic output.

    Returns:
        4-character hexadecimal digest of the hash.
    """
    hash_input = orjson.dumps(inputs, option=orjson.OPT_SORT_KEYS)
    hash_value = shake_256(hash_input).hexdigest(length=4)
    return hash_value


def compute_step_hash(nodes: ListConfig, name: str) -> str:
    """Compute hash from multiple OmegaConf nodes.

    Args:
        nodes: List of OmegaConf nodes to process.

    Returns:
        Hash generated from the combined input data of all nodes
        converted to dicts.
    """
    if name:
        return ""
    else:
        combined_inputs = []

        for node in nodes:
            if isinstance(node, str): # This element is a hash
                combined_inputs.append(node)
            elif isinstance(node, Container):
                node_dict = OmegaConf.to_container(node, resolve=True)
                combined_inputs.append(node_dict)

        return generate_hash(combined_inputs)


def compute_loader_hash(steps: ListConfig) -> str:
    """Compute hash from step hashes of multiple OmegaConf nodes.

    Args:
        steps: List of OmegaConf nodes containing step_hash attributes.
            Each step's step_hash is collected and used as input for the
            final hash.

    Returns:
        Hash generated from the combined step hashes of all provided steps.
    """
    combined_inputs = []

    for hash in steps:
        combined_inputs.append(hash)

    return generate_hash(combined_inputs)


def symlink(target: str, run_dir: str) -> None:
    """
    Symlinks a target file or directory to the specified run directory as relative.

    Args:
        target: The path of the file or directory to be symlinked.
        run_dir: The path to the run directory where the symlink will be created.
    """
    symlink_path = os.path.join(run_dir, os.path.basename(target))
    if not os.path.exists(target):
        logger.error(
            f"Target path `{target}` does not exist.\nError in preprocessing?"
        )
    elif os.path.exists(symlink_path):
        logger.warning(f"Symlink already exists: `{symlink_path}`, skipping.")
    else:
        logger.info(f"Symlinking:\n\t`{target}` -> `{symlink_path}`")
        # Compute relative path from symlink location to target
        relative_target = os.path.relpath(target, start=run_dir)
        os.symlink(relative_target, symlink_path)


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
