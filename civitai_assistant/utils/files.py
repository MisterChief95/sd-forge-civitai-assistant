import hashlib
import os
import json
from pathlib import Path
from typing import Any

from threading import Lock

from cachetools import TTLCache, cached
from cachetools.keys import hashkey

from civitai_assistant.const import PREVIEW_PNG, JSON
from civitai_assistant.utils.errors import get_exception_msg
from civitai_assistant.utils.logger import logger
from civitai_assistant.types import MetadataDescriptor, ModelDescriptor

try:
    from modules import hashes as _webui_hashes
except Exception:
    _webui_hashes = None

from civitai_assistant.utils.sd_path import get_checkpoint_dirs, get_lora_dirs, get_embeddings_dirs


# Map of base directory → cache-key prefix, matching hashes.py conventions.
# Evaluated lazily so cmd_opts-based custom dirs are included.
def _get_hash_prefix_map() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for d in get_checkpoint_dirs():
        pairs.append((d, "checkpoint"))
    for d in get_lora_dirs():
        pairs.append((d, "lora"))
    for d in get_embeddings_dirs():
        pairs.append((d, "textual_inversion"))
    return pairs


def _hash_cache_title(file_path: str) -> str:
    """Return the cache title used by the WebUI hash cache for a given model file."""
    abs_path = os.path.abspath(file_path)
    for base_dir, prefix in _get_hash_prefix_map():
        try:
            rel = os.path.relpath(abs_path, base_dir)
            if not rel.startswith(".."):
                return f"{prefix}/{rel}"
        except ValueError:
            # relpath raises ValueError on Windows when paths are on different drives
            pass
    # Fallback: use a stable, collision-resistant key for paths outside known model dirs.
    return f"external/{hashlib.sha256(abs_path.encode('utf-8')).hexdigest()[:16]}"


def _calculate_hash_direct(file_path: str, buffer_size: int = 8192) -> str:
    """Compute SHA-256 directly without using the WebUI cache."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(buffer_size):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def calculate_hash(file_path_str: str, buffer_size: int = 8192) -> str:
    """
    Returns the SHA-256 hash for a model file, delegating to the WebUI's
    built-in hash cache (modules.hashes) when available.  If the cache
    already holds a valid hash for the file it is returned immediately;
    otherwise the hash is computed, stored in the cache, and returned.

    Args:
        file_path (str): The path to the file to hash.
        buffer_size (int, optional): Buffer size used for direct computation
            when the WebUI cache module is unavailable. Defaults to 8192.
    Returns:
        str: The SHA-256 hash of the file in hexadecimal format.
    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
    """
    if not os.path.isfile(file_path_str):
        raise FileNotFoundError(f"The file {file_path_str} does not exist.")
    
    if _webui_hashes is not None:
        title = _hash_cache_title(file_path_str)
        result = _webui_hashes.sha256(Path(file_path_str), title)
        if result is not None:
            logger.info(f"Hash (cached): {os.path.basename(file_path_str)}")
            return result
        # sha256() returns None only when cmd_opts.no_hashing is set; fall through.

    logger.info(f"Computing hash: {os.path.basename(file_path_str)}")
    return _calculate_hash_direct(file_path_str, buffer_size)


def preview_exists(file_path: str) -> bool:
    """
    Checks if a given descriptor has a preview image.
    Args:
        descriptor (MetadataDescriptor): An object containing metadata information.
    Returns:
        bool: True if the descriptor has a preview image, False otherwise.
    """

    return os.path.exists(os.path.splitext(file_path)[0] + PREVIEW_PNG)


def write_preview(file_path: str, img_bytes: bytes) -> None:
    """
    Gets the input file's corresponding preview image file.
    Args:
        source (ModelDescriptor | str): An object containing metadata information or a file path.
    Returns:
        str: The path to the preview image file.
    """

    try:
        with open(os.path.splitext(file_path)[0] + PREVIEW_PNG, "wb") as img_file:
            img_file.write(img_bytes)

    except Exception as e:
        logger.error(f"Failed to write preview image: {get_exception_msg(e)}")


def has_json(file_path: str) -> bool:
    """
    Checks if a given descriptor has a JSON metadata file.
    Args:
        descriptor (MetadataDescriptor): An object containing metadata information.
    Returns:
        bool: True if the descriptor has a JSON metadata file, False otherwise.
    """

    return os.path.exists(os.path.splitext(file_path)[0] + JSON)


def to_json_file(file_path: str) -> str:
    """
    Gets the input files corresponding JSON metadata file.
    Args:
        descriptor (MetadataDescriptor): An object containing metadata information.
    Returns:
        bool: True if the descriptor has a JSON metadata file, False otherwise.
    """

    return os.path.splitext(file_path)[0] + JSON


def write_json_file(descriptor: ModelDescriptor) -> None:
    """
    Writes the metadata of a given descriptor to a JSON file.
    Args:
        descriptor (MetadataDescriptor | ModelDescriptor): The descriptor containing metadata to be written.
            If a ModelDescriptor is provided, its metadata_descriptor attribute is used.
    Returns:
        None
    """

    with open(to_json_file(descriptor.filename), "w") as json_file:
        json.dump(
            descriptor.metadata_descriptor.model_dump(by_alias=True),
            json_file,
            indent=4,
        )


def __cache_key(*args, **_) -> Any:
    return hashkey(args[0])


@cached(cache=TTLCache(maxsize=32, ttl=300), key=__cache_key, lock=Lock())
def generate_model_descriptor(
    model_file: str, recalculate_hash: bool = False
) -> ModelDescriptor:
    """
    Generates a model descriptor for the given model file.
    This function creates a `ModelDescriptor` object for the specified model file.
    It checks for an existing JSON file with metadata and validates it. If the JSON
    file does not exist or if the hash needs to be recalculated, it computes the hash
    of the model file. The resulting `ModelDescriptor`'s `MetadataDescriptor` is then
    written to a JSON file to avoid recomputing the hash in the future.
    Args:
        model_file (str): The path to the model file.
        recalculate_hash (bool, optional): Whether to recalculate the hash even if it exists. Defaults to False.
    Returns:
        ModelDescriptor: The generated model descriptor.
    """

    json_file: str = os.path.splitext(model_file)[0] + JSON

    if not os.path.exists(json_file):
        metadata_descriptor = MetadataDescriptor(hash=calculate_hash(model_file))
    else:
        with open(json_file, "r") as f:
            metadata_descriptor = MetadataDescriptor.model_validate(json.load(f))
            if not metadata_descriptor.hash or recalculate_hash:
                metadata_descriptor.hash = calculate_hash(model_file)

    model_descriptor = ModelDescriptor(
        metadata_descriptor=metadata_descriptor, filename=model_file
    )

    # Write the file so we don't have to recompute the hash
    write_json_file(model_descriptor)

    return model_descriptor


def filter_model_descriptors(
    model_descriptors: list[ModelDescriptor], filter_fn, overwrite_existing=False
):
    """
    Filter model descriptors based on a provided filter function.

    Args:
        descriptors (list[ModelDescriptor]): List of model descriptors to filter.
        filter_fn (callable, optional): A function that takes a ModelDescriptor and returns
            a boolean indicating whether to include it. If None, default_filter is used.
        default_filter (callable, optional): Default filter function used when filter_fn is None.
        overwrite_existing (bool, optional): If True, all descriptors are returned regardless of filter.
            Defaults to False.

    Returns:
        list[ModelDescriptor]: Filtered list of model descriptors.
    """
    if overwrite_existing:
        return model_descriptors

    return [d for d in model_descriptors if filter_fn(d)]
