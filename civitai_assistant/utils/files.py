import hashlib
import os
import json
from typing import Any

from threading import Lock

from cachetools import TTLCache, cached
from cachetools.keys import hashkey

from civitai_assistant.const import PREVIEW_PNG, JSON
from civitai_assistant.utils.errors import get_exception_msg
from civitai_assistant.utils.logger import logger
from civitai_assistant.type import MetadataDescriptor, ModelDescriptor


def calculate_hash(file_path: str) -> str:
    """
    Computes the SHA-256 hash of a file.
    Args:
        file_path (str): The path to the file to hash.
        buffer_size (int, optional): The size of the buffer to use when reading the file. Defaults to 8192.
    Returns:
        str: The SHA-256 hash of the file in hexadecimal format.
    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as file:
        while chunk := file.read(8192):
            sha256_hash.update(chunk)

    logger.info(f"Computed hash: {os.path.basename(file_path)}")

    return sha256_hash.hexdigest()


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
        json.dump(descriptor.metadata_descriptor.model_dump(by_alias=True), json_file, indent=4)


def __cache_key(*args, **_) -> Any:
    return hashkey(args[0])


@cached(cache=TTLCache(maxsize=32, ttl=300), key=__cache_key, lock=Lock())
def generate_model_descriptor(model_file: str, recalculate_hash: bool = False) -> ModelDescriptor:
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

    model_descriptor = ModelDescriptor(metadata_descriptor=metadata_descriptor, filename=model_file)

    # Write the file so we don't have to recompute the hash
    write_json_file(model_descriptor)

    return model_descriptor
