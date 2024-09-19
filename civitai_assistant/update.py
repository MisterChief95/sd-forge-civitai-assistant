import os
import time

import gradio as gr

from bs4 import BeautifulSoup as soup

import civitai_assistant.api as rest
from civitai_assistant.logger import logger
from civitai_assistant.type import ModelDescriptor, ModelType, CivitaiModel
from civitai_assistant.ui import progressify_sequence
from civitai_assistant.utils import files as file_utils


from modules.extra_networks import parse_prompt


def fetch_previews(model_previews: list[tuple[ModelDescriptor, str]]) -> list[tuple[ModelDescriptor, bytes]]:
    """
    Fetches previews for a list of model descriptors and their corresponding image URLs.
    Args:
        model_previews (list[tuple[ModelDescriptor, str]]): A list of tuples where each tuple contains a ModelDescriptor and a string representing the image URL.
    Returns:
        list[tuple[ModelDescriptor, bytes]]: A list of tuples where each tuple contains a ModelDescriptor and a Result object wrapping the response from the image URL request.
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


def update_metadata(
    modelTypes: list[ModelType], overwrite_existing: bool, recalculate_hash: bool, pr=gr.Progress()
) -> None:
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

    pr(0.1, "Finding model files")
    model_files: list[str] = file_utils.find_model_files(modelTypes)

    if not model_files:
        logger.info("No model files found. Exiting update process.")
        pr(1.0, "No model files found")
        time.sleep(1.5)
        return

    if overwrite_existing:
        pr(0.2, "Checking existing JSON metadata files")
        model_files = [file for file in model_files if file_utils.has_json(file)]

    if not model_files:
        logger.info("No model files found after filtering existing JSON metadata files. Exiting update process.")
        pr(1.0, "Done")
        time.sleep(1.5)
        return

    for model_file, progress in progressify_sequence(model_files, lower_bound=0.2, upper_bound=0.9):
        pr(progress, f"Building model descriptor: {os.path.basename(model_file)}")

        descriptor: ModelDescriptor = file_utils.generate_model_descriptor(model_file, recalculate_hash)

        if not descriptor:
            logger.error(f"Failed to build model descriptor for {os.path.basename(model_file)}")
            continue

        file_basename = os.path.basename(descriptor.filename)

        pr(progress, f"Fetching metadata: {file_basename}")
        civitai_model = rest.fetch_model_data(descriptor.metadata_descriptor.hash)

        if not civitai_model:
            logger.error(f"Failed to retrieve metadata for {file_basename}")
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

        pr(progress, f"Writing metadata: {file_basename}")
        try:
            file_utils.write_json_file(descriptor)
            logger.info(f"Updated metadata: {file_basename}")

        except Exception as e:
            logger.error(f"Failed to write metadata to JSON file: {str(e)}")

    pr(1.0, "Done")
    time.sleep(1.5)


def update_preview_images(
    modelTypes: list[ModelType], overwrite_existing: bool, recalculate_hash: bool, pr=gr.Progress()
) -> None:
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

    pr(0.1, "Finding model files")
    model_files: list[str] = file_utils.find_model_files(modelTypes)

    if not model_files:
        logger.info("No model files found. Exiting update process.")
        pr(1.0, "No model files found")
        time.sleep(1.5)
        return

    if overwrite_existing:
        pr(0.2, "Checking existing preview images")
        model_files = [file for file in model_files if file_utils.preview_exists(file)]

    if not model_files:
        logger.info("No model files found after filtering existing previews. Exiting update process.")
        pr(1.0, "Done")
        time.sleep(1.5)
        return

    for model_file, progress in progressify_sequence(model_files, lower_bound=0.2, upper_bound=0.9):
        pr(progress, f"Building model descriptor: {os.path.basename(model_file)}")

        descriptor: ModelDescriptor = file_utils.generate_model_descriptor(model_file, recalculate_hash)

        if not descriptor:
            logger.error(f"Failed to build model descriptor for {os.path.basename(model_file)}")
            continue

        file_basename = os.path.basename(descriptor.filename)

        pr(progress, f"Fetching metadata: {file_basename}")
        civitai_model = rest.fetch_model_data(descriptor.metadata_descriptor.hash)

        if not civitai_model:
            logger.error(f"Failed to retrieve metadata for {file_basename}")
            continue
        elif not civitai_model.images:
            logger.error(f"No preview image found for {file_basename}")
            continue

        pr(progress, f"Fetching image: {file_basename}")
        img_bytes: bytes = rest.fetch_image_preview(civitai_model.images[0].url)

        if img_bytes:
            file_utils.write_preview(descriptor, img_bytes)
            logger.info(f"Updated preview image for {os.path.basename(descriptor.filename)}")
        else:
            logger.warning(f"Failed to retrieve preview image for {os.path.basename(descriptor.filename)}")

    pr(1.0, "Done")
    time.sleep(1.5)
