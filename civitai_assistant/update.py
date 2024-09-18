import os
from collections.abc import Callable

from bs4 import BeautifulSoup as soup

import civitai_assistant.api as rest
from civitai_assistant.logger import logger
from civitai_assistant.type import ModelDescriptor, ModelType, CivitaiModel
from civitai_assistant.utils import files as file_utils

from modules import shared
from modules.shared import cmd_opts
from modules import sd_models
from modules.extra_networks import parse_prompt


def get_model_descriptors(model_types: list[ModelType], recalculate_hash: bool) -> list[ModelDescriptor]:
    """
    Generates a list of model descriptors for the given model types.
    Args:
        model_types (list[ModelType]): A list of model types for which to generate descriptors.
    Returns:
        list[ModelDescriptor]: A list of model descriptors corresponding to the provided model types.
    The function maps each model type to its respective directory using a predefined mapping.
    It then builds and collects model descriptors for each directory. If a model type is unknown
    or not selected, it prints a message and skips that type.
    """

    model_descriptors: list[ModelDescriptor] = []

    model_dir_map: dict[ModelType, Callable[[], str]] = {
        ModelType.CHECKPOINT: lambda: os.path.abspath(shared.cmd_opts.ckpt_dir or sd_models.model_path),
        ModelType.LORA: lambda: os.path.abspath(cmd_opts.lora_dir),
        ModelType.TEXTUAL_INVERSION: lambda: os.path.abspath(cmd_opts.embeddings_dir),
    }

    for modelType in model_types:
        model_dir: str = model_dir_map.get(modelType, lambda: None)()
        if model_dir is None:
            logger.warning(f"Unknown or unselected model type: {modelType}")
            continue

        model_descriptors.extend(file_utils.generate_model_descriptors(model_dir, recalculate_hash))

    logger.debug(f"Found {len(model_descriptors)} model descriptors")

    return model_descriptors


def fetch_previews(model_previews: list[tuple[ModelDescriptor, str]]) -> list[tuple[ModelDescriptor, bytes]]:
    """
    Fetches previews for a list of model descriptors and their corresponding image URLs.
    Args:
        model_previews (list[tuple[ModelDescriptor, str]]): A list of tuples where each tuple contains a ModelDescriptor and a string representing the image URL.
    Returns:
        list[tuple[ModelDescriptor, Result[requests.Response]]]: A list of tuples where each tuple contains a ModelDescriptor and a Result object wrapping the response from the image URL request.
    """

    return [(descriptor, rest.fetch_image_preview(img_url)) for descriptor, img_url in model_previews]


def fetch_metadata(model_descriptors: list[ModelDescriptor]) -> list[tuple[ModelDescriptor, CivitaiModel]]:
    """
    Fetch metadata for a list of model descriptors.
    Args:
        model_descriptors (list[ModelDescriptor]): A list of ModelDescriptor objects for which metadata needs to be fetched.
    Returns:
        list[tuple[ModelDescriptor, Result[CivitaiModel]]]: A list of tuples where each tuple contains a ModelDescriptor and the corresponding fetched metadata result.
    """

    return [
        (descriptor, rest.fetch_model_data(descriptor.metadata_descriptor.hash)) for descriptor in model_descriptors
    ]


def update_metadata(modelTypes: list[ModelType], overwrite_existing: bool, recalculate_hash: bool) -> None:
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

    model_descriptors: list[ModelDescriptor] = get_model_descriptors(modelTypes, recalculate_hash)

    if not model_descriptors:
        logger.warning("No models to update found. Exiting update process.")
        return

    if not overwrite_existing:
        model_descriptors = [
            descriptor for descriptor in model_descriptors if not file_utils.preview_exists(descriptor)
        ]

    if not model_descriptors:
        logger.info("No models to update found without metadata. Exiting update process.")
        return

    for descriptor, civitai_model in fetch_metadata(model_descriptors):
        if not civitai_model:
            logger.error(f"Failed to retrieve metadata for {os.path.basename(descriptor.filename)}")
            continue

        descriptor.metadata_descriptor.id = civitai_model.id
        descriptor.metadata_descriptor.modelId = civitai_model.modelId
        descriptor.metadata_descriptor.sd_version = (
            civitai_model.baseModel if civitai_model.baseModel != "Pony" else "Other"
        )

        activation_text: str = ", ".join(civitai_model.trainedWords) if civitai_model.trainedWords else ""
        if activation_text:
            activation_text = parse_prompt(activation_text)[0]

        descriptor.metadata_descriptor.activation_text = activation_text

        if civitai_model.description and not civitai_model.description.isspace():
            descriptor.metadata_descriptor.description = soup(civitai_model.description, "html.parser").get_text()

        try:
            file_utils.write_json_file(descriptor)
            logger.info(f"Updated metadata for {os.path.basename(descriptor.filename)}")

        except Exception as e:
            logger.error(f"Failed to write metadata to JSON file: {str(e)}")


def update_preview_images(modelTypes: list[ModelType], overwrite_existing: bool, recalculate_hash: bool) -> None:
    """
    Updates the preview image for a given model descriptor by calling the Civitai API.
    Args:
        model_descriptor (ModelDescriptor): The descriptor of the model for which the preview image is to be updated.
    Returns:
        None
    Raises:
        requests.exceptions.RequestException: If there is an issue with the HTTP request.
    Notes:
        - If the API call fails or the image retrieval fails, an error message is printed.
    """

    model_descriptors: list[ModelDescriptor] = get_model_descriptors(modelTypes, recalculate_hash)

    if not model_descriptors:
        logger.warning("No models to update found. Exiting update process.")
        return

    if not overwrite_existing:
        model_descriptors = [
            descriptor for descriptor in model_descriptors if not file_utils.preview_exists(descriptor)
        ]

    if not model_descriptors:
        logger.info("No models to update found without preview images. Exiting update process.")
        return

    model_img_url_list = [
        (descriptor, civitai_model.images[0].url)
        for descriptor, civitai_model in fetch_metadata(model_descriptors)
        if civitai_model and civitai_model.images
    ]

    for descriptor, img_bytes in fetch_previews(model_img_url_list):
        if img_bytes:
            file_utils.write_preview(descriptor, img_bytes)
            logger.info(f"Updated preview image for {os.path.basename(descriptor.filename)}")
        else:
            logger.warning(f"Failed to retrieve preview image for {os.path.basename(descriptor.filename)}")
