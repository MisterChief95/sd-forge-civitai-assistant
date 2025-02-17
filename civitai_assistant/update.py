import os
import time
from typing import Optional

import gradio as gr

from bs4 import BeautifulSoup as soup

import civitai_assistant.api as rest
import civitai_assistant.utils.files as files
import civitai_assistant.utils.sd_path as sd_path
from civitai_assistant.utils.logger import logger
from civitai_assistant.types import CivitaiModel, ModelDescriptor, ModelType
from civitai_assistant.ui import progressify_sequence

from modules.extra_networks import parse_prompt


BUILD_DESCRIPTOR: str = "Building model descriptor: {0}"
CHECK_OVERWRITE: str = "Checking for overwrite"
FAILED_BUILD_DESCRIPTOR: str = "Failed to build model descriptor for {0}"
FAILED_META: str = "Failed to retrieve metadata for {0}"
FETCHING_META: str = "Fetching metadata: {0}"
FINDING_MODELS: str = "Finding model files"
NO_MODELS_AFTER_FILTER: str = "No model files found after filtering"
NO_MODELS_FOUND: str = "No model files found"


def update_metadata(
    modelTypes: list[ModelType],
    overwrite_existing: bool,
    recalculate_hash: bool,
    pr: gr.Progress = gr.Progress(),  # noqa: B008
) -> None:
    """
    Updates the metadata for a list of model files.
    Args:
        modelTypes (list[ModelType]): A list of model types to search for.
        overwrite_existing (bool): If True, existing metadata will be overwritten.
        recalculate_hash (bool): If True, the hash for each model will be recalculated.
        pr (gr.Progress, optional): A progress indicator. Defaults to gr.Progress().
    Returns:
        None
    """

    pr(0.1, "Finding model files")
    model_files: list[str] = sd_path.find_model_files(modelTypes)

    if not model_files:
        logger.info(NO_MODELS_FOUND)
        gr.Info(NO_MODELS_FOUND)
        pr(1.0, NO_MODELS_FOUND)
        time.sleep(1.5)
        return

    pr(0.2, "Checking for overwrite")
    if not overwrite_existing:
        model_files = [file for file in model_files if not files.has_json(file)]

    if not model_files:
        logger.info(NO_MODELS_AFTER_FILTER)
        gr.Info(NO_MODELS_FOUND)
        pr(1.0, "Done")
        time.sleep(1.5)
        return

    for model_file, progress in progressify_sequence(
        model_files, lower_bound=0.2, upper_bound=0.9
    ):
        pr(progress, BUILD_DESCRIPTOR.format(os.path.basename(model_file)))

        descriptor: ModelDescriptor = files.generate_model_descriptor(
            model_file, recalculate_hash
        )

        if not descriptor:
            msg = FAILED_BUILD_DESCRIPTOR.format(os.path.basename(model_file))
            logger.error(msg)
            gr.Warning(msg)
            continue

        pr(progress, FETCHING_META.format(descriptor.file_basename))

        civitai_model: Optional[CivitaiModel] = rest.fetch_by_hash(
            descriptor.metadata_descriptor.hash
        )
        description = (
            rest.fetch_model_description(civitai_model.modelId) if civitai_model else ""
        )

        if not civitai_model:
            msg = FAILED_META.format(descriptor.file_basename)
            logger.error(FAILED_META.format(descriptor.file_basename))
            continue

        descriptor.metadata_descriptor.model_id = civitai_model.modelId
        descriptor.metadata_descriptor.sd_version = (
            civitai_model.baseModel
            if civitai_model.baseModel or civitai_model.baseModel != "Pony"
            else "Other"
        )

        activation_text: str = (
            ", ".join(civitai_model.trainedWords) if civitai_model.trainedWords else ""
        )
        if activation_text:
            activation_text = parse_prompt(activation_text)[0]

        descriptor.metadata_descriptor.activation_text = activation_text

        if description and not description.isspace():
            # TODO: Add HTML support
            # if opts.ca_use_html_descriptions:
            descriptor.metadata_descriptor.description = soup(
                description, "html.parser"
            ).get_text()
            # else:
            #     descriptor.metadata_descriptor.description = description

        pr(progress, f"Writing metadata: {descriptor.file_basename}")
        try:
            files.write_json_file(descriptor)
            logger.info(f"Updated metadata: {descriptor.file_basename}")

        except Exception as e:
            logger.error(f"Failed to write metadata to JSON file: {str(e)}")

    pr(1.0, "Done")
    time.sleep(1.5)


def update_preview_images(
    modelTypes: list[ModelType],
    overwrite_existing: bool,
    recalculate_hash: bool,
    pr: gr.Progress = gr.Progress(),  # noqa: B008
) -> None:
    """
    Updates the preview images for the given model types.
    Args:
        modelTypes (list[ModelType]): A list of model types to update preview images for.
        overwrite_existing (bool): If True, overwrite existing preview images.
        recalculate_hash (bool): If True, recalculate the hash for the model files.
        pr (gr.Progress, optional): A progress indicator. Defaults to gr.Progress().
    Returns:
        None
    """

    pr(0.1, FINDING_MODELS)
    model_files: list[str] = sd_path.find_model_files(modelTypes)

    if not model_files:
        logger.info(NO_MODELS_FOUND)
        gr.Info(NO_MODELS_FOUND)
        pr(1.0, NO_MODELS_FOUND)
        time.sleep(1.5)
        return

    pr(0.2, CHECK_OVERWRITE)
    if not overwrite_existing:
        model_files = [file for file in model_files if not files.preview_exists(file)]

    if not model_files:
        logger.info(NO_MODELS_AFTER_FILTER)
        gr.Info(NO_MODELS_FOUND)
        pr(1.0, "Done")
        time.sleep(1.5)
        return

    for model_file, progress in progressify_sequence(
        model_files, lower_bound=0.2, upper_bound=0.9
    ):
        pr(progress, BUILD_DESCRIPTOR.format(os.path.basename(model_file)))

        descriptor: ModelDescriptor = files.generate_model_descriptor(
            model_file, recalculate_hash
        )

        if not descriptor:
            logger.error(FAILED_BUILD_DESCRIPTOR.format(os.path.basename(model_file)))
            continue

        pr(progress, FETCHING_META.format(descriptor.file_basename))

        civitai_model: Optional[CivitaiModel] = rest.fetch_by_hash(
            descriptor.metadata_descriptor.hash
        )

        if not civitai_model or not civitai_model.images:
            msg = f"Failed to retrieve metadata or no preview image found for {descriptor.file_basename}"

            logger.warning(msg)
            gr.Warning(msg)
            continue

        pr(progress, f"Fetching image: {descriptor.file_basename}")

        first_image = next(filter(lambda image_data: image_data.type == "image", civitai_model.images), None)
        if not first_image or not first_image.url:
            logger.warning(f"No image found for {descriptor.file_basename}")
            continue

        img_bytes = rest.fetch_image_preview(first_image.url)

        if img_bytes:
            files.write_preview(descriptor.filename, img_bytes)
            logger.info(f"Updated preview image for {descriptor.file_basename}")
        else:
            logger.warning(
                f"Failed to retrieve preview image for {descriptor.file_basename}"
            )

    pr(1.0, "Done")
    time.sleep(1.5)
