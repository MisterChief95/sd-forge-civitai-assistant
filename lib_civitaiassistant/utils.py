import hashlib
import os
import json

from .types import MetadataDescriptor


def build_file_hash(file_path: str, buffer_size: int = 8192) -> str:
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

    print(f"Computing hash for {os.path.basename(file_path)}", end="... ", flush=True)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as file:
        while chunk := file.read(buffer_size):
            sha256_hash.update(chunk)

    print("done", flush=True)

    return sha256_hash.hexdigest()

def write_metadata_json(descriptor: MetadataDescriptor, filename: str) -> None:
    """
    Writes the metadata of a given descriptor to a JSON file.
    Args:
        descriptor (MetadataDescriptor): An object containing metadata information.
    The JSON file is created in the same directory as the descriptor's filename,
    with the same base name and a .json extension.
    """

    json_path = os.path.splitext(filename)[0] + ".json"
    with open(json_path, "w") as json_file:
        json.dump(descriptor.model_dump(), json_file, indent=4)
