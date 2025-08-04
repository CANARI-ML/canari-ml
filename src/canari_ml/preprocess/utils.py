import logging
import os
from hashlib import shake_256
from types import SimpleNamespace

logger = logging.getLogger(__name__)
import orjson
from omegaconf import Container, ListConfig, OmegaConf


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
    Symlinks a target file or directory to the specified run directory.

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
        os.symlink(target, symlink_path)
