import hashlib
import os
import json

from .types import MetadataDescriptor, ModelDescriptor


def calculate_hash(file_path: str, buffer_size: int = 8192) -> str:
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


def preview_exists(source: ModelDescriptor | str) -> bool:
    """
    Checks if a given descriptor has a preview image.
    Args:
        descriptor (MetadataDescriptor): An object containing metadata information.
    Returns:
        bool: True if the descriptor has a preview image, False otherwise.
    """

    file_path: str = source.filename if isinstance(source, ModelDescriptor) else source

    return os.path.exists(os.path.splitext(file_path)[0] + ".preview.png")


def write_preview(source: ModelDescriptor | str, img_bytes: bytes) -> None:
    """
    Gets the input file's corresponding preview image file.
    Args:
        source (ModelDescriptor | str): An object containing metadata information or a file path.
    Returns:
        str: The path to the preview image file.
    """

    file_path: str = source.filename if isinstance(source, ModelDescriptor) else source

    img_path: str = os.path.splitext(file_path)[0] + ".preview.png"
    with open(img_path, "wb") as img_file:
        img_file.write(img_bytes)


def has_json(source: ModelDescriptor | str) -> bool:
    """
    Checks if a given descriptor has a JSON metadata file.
    Args:
        descriptor (MetadataDescriptor): An object containing metadata information.
    Returns:
        bool: True if the descriptor has a JSON metadata file, False otherwise.
    """

    file_path: str = source.filename if isinstance(source, ModelDescriptor) else source

    return os.path.exists(os.path.splitext(file_path)[0] + ".json")


def to_json_file(source: ModelDescriptor | str) -> bool:
    """
    Gets the input files corresponding JSON metadata file.
    Args:
        descriptor (MetadataDescriptor): An object containing metadata information.
    Returns:
        bool: True if the descriptor has a JSON metadata file, False otherwise.
    """

    file_path: str = source.filename if isinstance(source, ModelDescriptor) else source

    return os.path.splitext(file_path)[0] + ".json"


def write_json_file(source: ModelDescriptor) -> None:
    """
    Writes the metadata of a given descriptor to a JSON file.
    Args:
        descriptor (MetadataDescriptor): An object containing metadata information.
    The JSON file is created in the same directory as the descriptor's filename,
    with the same base name and a .json extension.
    """

    with open(to_json_file(source), "w") as json_file:
        json.dump(source.metadata_descriptor.model_dump(by_alias=True), json_file, indent=4)


def generate_model_descriptors(model_dir: str) -> list[ModelDescriptor]:
    """
    Builds a list of model descriptors from the given directory.
    This function walks through the specified directory to find all files with the
    ".safetensors" extension. For each found file, it attempts to locate a corresponding
    JSON file containing metadata. If the JSON file is not found, it generates a new
    metadata descriptor by computing the file's hash. If the JSON file is found, it
    validates and loads the metadata. If the loaded metadata does not contain a hash,
    it computes and assigns the hash.
    Args:
        model_dir (str): The directory containing the model files and their metadata.
    Returns:
        list[ModelDescriptor]: A list of ModelDescriptor objects containing the metadata
        and filenames of the models.
    """

    model_files = []
    for root, _, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".safetensors"):
                model_files.append(os.path.join(root, file))

    model_descriptors = []

    for model_file in model_files:
        json_file: str = os.path.splitext(model_file)[0] + ".json"
        json_path: str = os.path.join(model_dir, json_file)

        metadata_descriptor: MetadataDescriptor = None

        if not os.path.exists(json_path):
            metadata_descriptor = MetadataDescriptor(hash=calculate_hash(model_file))
        else:
            with open(json_path, "r") as f:
                metadata_descriptor = MetadataDescriptor.model_validate(json.load(f))
                if not metadata_descriptor.hash:
                    metadata_descriptor.hash = calculate_hash(model_file)

        # Immediately save the metadata to the JSON file
        write_json_file()

        model_descriptors.append(
            ModelDescriptor(
                metadata_descriptor=metadata_descriptor,
                filename=model_file,
                requires_write=not metadata_descriptor.id or not metadata_descriptor.modelId,
            )
        )

    return model_descriptors
