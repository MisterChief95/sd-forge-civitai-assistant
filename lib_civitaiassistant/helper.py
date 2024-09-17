import logging
import asyncio
import requests
import os
from collections.abc import Callable

from bs4 import BeautifulSoup as soup

import lib_civitaiassistant.file_utils as file_utils
import lib_civitaiassistant.rest as rest
from .types import ModelDescriptor, ModelType, CivitaiModel, Result

from modules import shared
from modules.shared import cmd_opts
from modules import sd_models


logger = logging.getLogger(__name__)


def get_model_descriptors(model_types: list[ModelType]) -> list[ModelDescriptor]:
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
            logger.error(f"Unknown or unselected model type: {modelType}")
            continue

        model_descriptors.extend(file_utils.build_model_descriptors_for_dir(model_dir))

    return model_descriptors


async def fetch_previews_async(
    model_previews: list[tuple[ModelDescriptor, str]]
) -> list[tuple[ModelDescriptor, Result[requests.Response]]]:
    """
    Asynchronously fetches previews for a list of model descriptors and their corresponding image URLs.
    Args:
        model_previews (list[tuple[ModelDescriptor, str]]): A list of tuples where each tuple contains a ModelDescriptor and a string representing the image URL.
    Returns:
        list[tuple[ModelDescriptor, Result[requests.Response]]]: A list of tuples where each tuple contains a ModelDescriptor and a Result object wrapping the response from the image URL request.
    """

    tasks = []
    for descriptor, img_url in model_previews:
        tasks.append((descriptor, rest.send_request(img_url, stream=True)))

    await asyncio.gather(*tasks)


async def fetch_metadata_async(
    model_descriptors: list[ModelDescriptor],
) -> list[tuple[ModelDescriptor, Result[CivitaiModel]]]:
    """
    Asynchronously fetch metadata for a list of model descriptors.
    Args:
        model_descriptors (list[ModelDescriptor]): A list of ModelDescriptor objects for which metadata needs to be fetched.
    Returns:
        list[tuple[ModelDescriptor, Result[CivitaiModel]]]: A list of tuples where each tuple contains a ModelDescriptor and the corresponding fetched metadata result.
    """

    tasks = []
    for descriptor in model_descriptors:
        tasks.append((descriptor, rest.fetch_model_data(descriptor.metadata_descriptor.hash)))

    await asyncio.gather(*tasks)


def update_metadata(modelTypes: list[ModelType], overwrite_existing: bool) -> None:
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
    
    logger.info("Updating metadata for the following types: ", modelTypes)

    model_descriptors: list[ModelDescriptor] = get_model_descriptors(modelTypes)

    if not model_descriptors:
        logger.warning("No model descriptors found. Exiting update process.")
        return

    if not overwrite_existing:
        model_descriptors = [
            descriptor for descriptor in model_descriptors if not file_utils.has_preview_image(descriptor)
        ]

    api_model_list: list[tuple[ModelDescriptor, Result[CivitaiModel]]] = asyncio.run(
        fetch_metadata_async(model_descriptors)
    )

    for descriptor, result in api_model_list:
        if not result.value:
            logger.error(f"Failed to retrieve metadata for {os.path.basename(descriptor.filename)}")
            continue

        descriptor.metadata_descriptor.id = result.value.id
        descriptor.metadata_descriptor.modelId = result.value.modelId
        descriptor.metadata_descriptor.sd_version = (
            result.value.baseModel if result.value.baseModel != "Pony" else "Other"
        )
        descriptor.metadata_descriptor.activation_text = (
            ",, ".join(result.value.trainedWords) if result.value.trainedWords else ""
        )

        if result.value.description and not result.value.description.isspace():
            descriptor.metadata_descriptor.description = soup(result.value.description, "html.parser").get_text()

        file_utils.write_metadata_json(descriptor)

    logging.info("Metadata update completed")


def update_preview_images(modelTypes: list[ModelType], overwrite_existing: bool) -> None:
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
    
    logging.info("Updating preview images for the following types: ", modelTypes)

    model_descriptors: list[ModelDescriptor] = get_model_descriptors(modelTypes)

    if not model_descriptors:
        logging.warning("No model descriptors found. Exiting update process.")
        return

    if not overwrite_existing:
        model_descriptors = [
            descriptor for descriptor in model_descriptors if not file_utils.has_preview_image(descriptor)
        ]

    api_model_list: list[tuple[ModelDescriptor, Result[CivitaiModel]]] = asyncio.run(
        fetch_metadata_async(model_descriptors)
    )

    model_img_url_list = [
        (descriptor, result.value.images[0].url)
        for descriptor, result in api_model_list
        if result.value and result.value.images
    ]

    preview_image_list: list[tuple[ModelDescriptor, Result[requests.Response]]] = asyncio.run(
        fetch_previews_async(model_img_url_list)
    )

    for descriptor, img_result in preview_image_list:
        if img_result.value:
            file_utils.write_preview_image_file(descriptor, img_result.value.content)

    logging.info("Finished calling API")
