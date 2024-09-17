import asyncio
import json
import requests
import os

from bs4 import BeautifulSoup

from .utils import build_file_hash, write_metadata_json
from .types import ModelDescriptor, ModelType, MetadataDescriptor, CivitaiModel

from modules import shared
from modules.shared import cmd_opts
from modules import sd_models


def build_model_descriptors(model_dir: str) -> list[ModelDescriptor]:
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
            metadata_descriptor = MetadataDescriptor(hash=build_file_hash(model_file))
        else:
            with open(json_path, "r") as f:
                metadata_descriptor = MetadataDescriptor.model_validate(json.load(f))
                if not metadata_descriptor.hash:
                    metadata_descriptor.hash = build_file_hash(model_file)

        # Immediately save the metadata to the JSON file
        write_metadata_json(metadata_descriptor, model_file)

        model_descriptors.append(
            ModelDescriptor(
                metadata_descriptor=metadata_descriptor,
                filename=model_file,
            )
        )

    return model_descriptors


async def call_api(model_descriptor: ModelDescriptor) -> None:
    """
    Calls the Civitai API to retrieve model information and updates the given model descriptor with the retrieved data.
    Args:
        model_descriptor (ModelDescriptor): The descriptor of the model for which information is to be retrieved.
    Raises:
        requests.exceptions.RequestException: If there is an issue with the HTTP request.
    Side Effects:
        Updates the `metadata_descriptor` attribute of the `model_descriptor` with the retrieved model information.
        Saves the first image from the retrieved model information as a preview image in the same directory as the model file.
    Notes:
        - If the API call fails or the model information cannot be validated, an error message is printed.
        - If the image retrieval fails, an error message is printed.
    """

    print(f"Calling API for {os.path.basename(model_descriptor.filename)}", end="... ", flush=True)

    # Use first 10 characters of hash - AutoV2
    response: requests.Response = requests.get(
        url=f"https://civitai.com/api/v1/model-versions/by-hash/{model_descriptor.metadata_descriptor.hash[0:10]}",
        timeout=5,
    )

    if response.status_code != 200:
        print(f"Failed to get model info")
        return

    try:
        civitai_model = CivitaiModel.model_validate(response.json())

        if civitai_model is None:
            print(f"Failed to validate model info")
            return
    except Exception as e:
        print(f"Error occurred while validating model information for {model_descriptor.filename}: {e}")
        return

    model_descriptor.metadata_descriptor.id = civitai_model.id
    model_descriptor.metadata_descriptor.modelId = civitai_model.modelId
    model_descriptor.metadata_descriptor.sd_version = (
        civitai_model.baseModel if civitai_model.baseModel != "Pony" else "Other"
    )
    model_descriptor.metadata_descriptor.activation_text = (
        ",, ".join(civitai_model.trainedWords) if civitai_model.trainedWords else ""
    )

    if civitai_model.description and not civitai_model.description.isspace():
        model_descriptor.metadata_descriptor.description = BeautifulSoup(
            civitai_model.description, "html.parser"
        ).get_text()

    # save first image as preview
    img_response: requests.Response = requests.get(civitai_model.images[0].url, stream=True)

    if img_response.status_code != 200:
        print(f"Failed to get image for {model_descriptor.filename}")

    else:
        img_path: str = os.path.splitext(model_descriptor.filename)[0] + ".preview.png"
        with open(img_path, "wb") as img_file:
            img_file.write(img_response.content)

def update_metadata(modelTypes: list[ModelType]) -> None:
    """
    Updates the metadata for a list of model types by building model descriptors
    and calling an API for each descriptor. The metadata is then saved to a JSON file.
    Args:
        modelTypes (list[ModelType]): A list of model types to update metadata for.
    Returns:
        None
    Raises:
        None
    Notes:
        - The function determines the directory for each model type and builds model descriptors.
        - It then calls an API for each descriptor and saves the metadata to a JSON file.
        - If an unknown model type is encountered, it prints an error message.
        - If no model directory is selected, it prints an error message and returns.
    """
    print("Updating metadata for the following types: ", modelTypes)

    model_dir: str = None
    model_descriptors: dict[ModelType, list[ModelDescriptor]] = {}

    for modelType in modelTypes:
        match modelType:
            case ModelType.CHECKPOINT:
                model_dir = os.path.abspath(shared.cmd_opts.ckpt_dir or sd_models.model_path)
            case ModelType.LORA:
                model_dir = os.path.abspath(cmd_opts.lora_dir)
            case ModelType.TEXTUAL_INVERSION:
                model_dir = os.path.abspath(cmd_opts.embeddings_dir)
            case _:
                print("Unknown model type")

        if model_dir is None:
            print("No model options selected")
            return

        model_descriptors[modelType] = build_model_descriptors(model_dir)

    async def update_all_metadata():
        tasks = []
        for _, descriptors in model_descriptors.items():
            for descriptor in descriptors:
                tasks.append(call_api(descriptor))
        await asyncio.gather(*tasks)

    asyncio.run(update_all_metadata())

    print("Finished calling API")

    for _, descriptors in model_descriptors.items():
        for descriptor in descriptors:
            write_metadata_json(descriptor.metadata_descriptor, descriptor.filename)

    print("Finished updating metadata")
