import os
import time
import asyncio
from typing import Callable

import gradio as gr
from bs4 import BeautifulSoup as soup

import civitai_assistant.api as api
import civitai_assistant.utils.files as files
import civitai_assistant.utils.sd_path as sd_path
from civitai_assistant.utils.logger import logger
from civitai_assistant.types import (
    CivitaiModel,
    ModelDescriptor,
    ModelType,
    UpdateOptions,
    UpdateType,
)

from modules.extra_networks import parse_prompt
from modules.shared import opts


class ProgressWrapper:
    """
    A wrapper class that scales progress values between a start and end range.
    This class is used to map a progress fraction (0.0-1.0) to a specific range
    of progress values (start_progress to end_progress). This is useful when a
    process is a sub-step of a larger process and you want to report progress
    relative to the overall process.
    Parameters:
    ----------
    pr : callable
        The progress reporting function to be called with the scaled progress value.
    start_progress : float
        The starting value of the progress range (corresponds to fraction=0.0).
    end_progress : float
        The ending value of the progress range (corresponds to fraction=1.0).
    Methods:
    -------
    __call__(fraction=None, description="")
        Report progress by scaling the fraction to the defined progress range.
        Parameters:
        fraction : float, optional
            Current progress as a value between 0.0 and 1.0. If None, uses start_progress.
        description : str, optional
            Description text to accompany the progress update.
    """

    def __init__(self, pr, start_progress, end_progress):
        self.start_progress = start_progress
        self.end_progress = end_progress
        self.pr = pr

    def __call__(self, fraction=None, description=""):
        if fraction is None:
            scaled_progress = self.start_progress
        else:
            scaled_progress = self.start_progress + (
                fraction * (self.end_progress - self.start_progress)
            )
        self.pr(scaled_progress, description)


def find_and_build_model_descriptors(
    model_types: list[ModelType],
    recalculate_hash: bool,
    pr: gr.Progress,
) -> list[ModelDescriptor]:
    """
    Find model files and build model descriptors.

    Args:
        model_types: Types of models to process
        overwrite_existing: Whether to overwrite existing files
        recalculate_hash: Whether to recalculate model hashes
        pr: Gradio progress component
        filter_fn: Optional function to filter model files

    Returns:
        List of ModelDescriptor objects
    """
    # Find model files
    pr(0.1, "Finding model files")
    model_files: list[str] = sd_path.find_model_files(model_types)

    if not model_files:
        logger.info("No models found")
        gr.Info("No models found")
        pr(fraction=1.0, description="No models found")
        time.sleep(1.5)
        return []

    # Build model descriptors
    pr(0.3, "Building model descriptors")
    model_descriptors = []
    for model_file in model_files:
        try:
            descriptor = files.generate_model_descriptor(model_file, recalculate_hash)
            model_descriptors.append(descriptor)
        except Exception as e:
            msg = f"Failed to build model descriptor for {os.path.basename(model_file)}: {str(e)}"
            logger.error(msg)
            gr.Warning(msg)

    return model_descriptors


async def process_models_async(
    model_descriptors: list[ModelDescriptor],
    processor_fn: Callable[[ModelDescriptor, CivitaiModel], None],
    pr: gr.Progress,
    batch_size: int = 5,
) -> None:
    """
    Process model files asynchronously with progress tracking.

    Args:
        model_types: Types of models to process
        overwrite_existing: Whether to overwrite existing files
        recalculate_hash: Whether to recalculate model hashes
        processor_fn: Function to process each model
        filter_fn: Optional function to filter model files
        batch_size: Number of models to process in parallel
        pr: Gradio progress component
    """
    api_key = opts.data.get("ca_api_key", None)

    if not model_descriptors:
        return

    # Process models in batches for better performance without overwhelming the API
    total_models = len(model_descriptors)
    for i in range(0, total_models, batch_size):
        batch = model_descriptors[i : i + batch_size]
        batch_len = len(batch)

        # Update progress
        progress_start = 0.3 + (i / total_models) * 0.6
        progress_end = 0.3 + ((i + batch_len) / total_models) * 0.6
        batch_progress = progress_start

        try:
            # Fetch model data for the batch
            pr(
                batch_progress,
                f"Fetching metadata for batch {i // batch_size + 1}/{(total_models - 1) // batch_size + 1}",
            )
            model_hashes = [d.metadata_descriptor.hash for d in batch]
            civitai_models = await api.fetch_multiple_by_hash(model_hashes, api_key)

            # Process each model in the batch
            for batch_index, descriptor in enumerate(batch):
                model_hash = descriptor.metadata_descriptor.hash
                civitai_model = civitai_models.get(model_hash)

                if batch_len == 1:
                    batch_progress = progress_end
                else:
                    batch_progress = progress_start + (
                        (batch_index / (batch_len - 1)) * (progress_end - progress_start)
                    )
                pr(batch_progress, f"Processing {descriptor.file_basename}")

                if not civitai_model:
                    logger.error(
                        f"Failed to retrieve metadata for {descriptor.file_basename}"
                    )
                    continue

                # Process the model
                try:
                    await processor_fn(descriptor, civitai_model)
                except Exception as e:
                    logger.error(
                        f"Error processing {descriptor.file_basename}: {str(e)}"
                    )
                    continue
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            continue

        # Give a small delay between batches to let resources clean up
        await asyncio.sleep(0.5)


async def metadata_processor_async(
    descriptor: ModelDescriptor,
    civitai_model: CivitaiModel,
) -> None:
    """Process a model to update its metadata asynchronously."""
    api_key = opts.data.get("ca_api_key", None)

    # Fetch additional description
    description = (
        await api.fetch_model_description(civitai_model.modelId, api_key)
        if civitai_model
        else ""
    )

    # Update descriptor with Civitai data
    descriptor.metadata_descriptor.model_id = civitai_model.modelId
    descriptor.metadata_descriptor.sd_version = (
        civitai_model.baseModel
        if civitai_model.baseModel and civitai_model.baseModel != "Pony"
        else "Other"
    )

    # Process activation text
    activation_text: str = (
        ", ".join(civitai_model.trainedWords) if civitai_model.trainedWords else ""
    )
    if activation_text:
        activation_text = parse_prompt(activation_text)[0]
    descriptor.metadata_descriptor.activation_text = activation_text

    # Process description
    use_html = opts.data.get("ca_use_html_descriptions", False)
    if description and not description.isspace():
        descriptor.metadata_descriptor.description = (
            description if use_html else soup(description, "html.parser").get_text()
        )
    elif descriptor.metadata_descriptor.description is None:
        # Mark as attempted-but-empty so the filter doesn't keep re-fetching
        descriptor.metadata_descriptor.description = ""

    # Write metadata to file
    try:
        files.write_json_file(descriptor)
        logger.info(f"Updated metadata: {descriptor.file_basename}")
    except Exception as e:
        logger.error(f"Failed to write metadata to JSON file: {str(e)}")


async def image_processor_async(
    descriptor: ModelDescriptor,
    civitai_model: CivitaiModel,
) -> None:
    """Process a model to update its preview image asynchronously."""
    if not civitai_model.images:
        msg = f"No preview image found for {descriptor.file_basename}"
        logger.warning(msg)
        gr.Warning(msg)
        return

    # Find first image
    first_image = next(
        filter(lambda image_data: image_data.type == "image", civitai_model.images),
        None,
    )
    if not first_image or not first_image.url:
        logger.warning(f"No image found for {descriptor.file_basename}")
        return

    # Download and save image
    img_bytes = await api.fetch_image_preview(first_image.url)
    if img_bytes:
        files.write_preview(descriptor.filename, img_bytes)
        logger.info(f"Updated preview image for {descriptor.file_basename}")
    else:
        logger.warning(
            f"Failed to retrieve preview image for {descriptor.file_basename}"
        )


# Synchronous wrapper functions for Gradio compatibility
def update_metadata(
    model_descriptors: list[ModelDescriptor],
    pr: gr.Progress,
) -> None:
    """Updates metadata for model files."""
    # Get the current event loop or create a new one if needed
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Run the async function without closing the loop afterward
    loop.run_until_complete(
        process_models_async(
            model_descriptors=model_descriptors,
            pr=pr,
            processor_fn=metadata_processor_async,
        )
    )


def update_preview_images(
    model_descriptors: list[ModelDescriptor],
    pr: gr.Progress,
) -> None:
    """Updates preview images for model files."""
    # Get the current event loop or create a new one if needed
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Run the async function without closing the loop afterward
    loop.run_until_complete(
        process_models_async(
            model_descriptors=model_descriptors,
            pr=pr,
            processor_fn=image_processor_async,
        )
    )


def process_update(
    update_type_name: str,
    model_descriptors: list,
    wrapped_pr,
    overwrite_existing: bool,
    filter_check_func,
    update_func,
    i: int,
    total_types: int,
):
    """
    Process updates for models based on the provided parameters.

    Args:
        update_type_name: Name of the update type (metadata or preview images)
        model_descriptors: List of model descriptors to process
        wrapped_pr: Progress reporting function
        overwrite_existing: Whether to overwrite existing files
        filter_check_func: Function to check if a model needs updates
        update_func: Function to call for updating models
        i: Current update type index
        total_types: Total number of update types
    """
    logger.info(f"Updating {update_type_name} ({i + 1}/{total_types})")

    wrapped_pr(fraction=0.2, description="Checking for overwrite")

    # Filter models that need updates
    filtered_descriptors = files.filter_model_descriptors(
        model_descriptors=model_descriptors,
        overwrite_existing=overwrite_existing,
        filter_fn=filter_check_func,
    )

    if not filtered_descriptors:
        gr.Info(f"No models need {update_type_name} updates")
        logger.info(f"No models need {update_type_name} updates")
    else:
        update_func(
            model_descriptors=filtered_descriptors,
            pr=wrapped_pr,
        )
    wrapped_pr(fraction=0.95, description="")


def _json_missing_essential_fields(descriptor):
    """Check if JSON metadata file is missing any essential fields."""
    if not files.has_json(descriptor.filename):
        return True

    try:
        md = descriptor.metadata_descriptor
        # model_id must be present; description must have been attempted (not None).
        # A None description means the fetch was never completed, so we re-process.
        # An empty string means the fetch was done but CivitAI returned nothing, which
        # is considered complete (avoids endless re-fetching for undescribed models).
        return not md.model_id or md.description is None

    except Exception:
        # If there's any issue reading the metadata, consider it missing fields
        return True


def update_models(
    model_types: list[ModelType],
    update_types: list[str],
    options: list[str],
    pr: gr.Progress,
) -> None:
    """
    Updates models with specified update types (metadata and/or preview images).

    Args:
        model_types: Types of models to process
        update_types: List of update types ('metadata', 'preview_images')
        options: List of update options
        pr: Gradio progress component
    """

    # Filter valid update types
    valid_update_types = {member.value for member in UpdateType.__members__.values()}
    update_types = [t for t in update_types if t in valid_update_types]

    if not update_types:
        gr.Warning("No valid update types selected")
        return

    overwrite_existing = UpdateOptions.OVERWRITE_EXISTING.value in options
    recalculate_hash = UpdateOptions.RECALCULATE_HASHES.value in options

    total_types = len(update_types)
    progress_per_type = 1.0 / total_types

    # Get model descriptors first
    model_descriptors = find_and_build_model_descriptors(
        model_types=model_types,
        recalculate_hash=recalculate_hash,
        pr=pr,
    )

    if not model_descriptors:
        gr.Warning("No models found to process")
        return

    wrapped_pr: ProgressWrapper = None

    for i, update_type in enumerate(update_types):
        # Calculate progress range for this update type
        start_progress = i * progress_per_type
        end_progress = (i + 1) * progress_per_type

        wrapped_pr = ProgressWrapper(pr, start_progress, end_progress)

        # Call the appropriate update function with filtered model descriptors
        if update_type == UpdateType.METADATA.value:            
            process_update(
                update_type_name="metadata",
                model_descriptors=model_descriptors,
                wrapped_pr=wrapped_pr,
                overwrite_existing=overwrite_existing,
                filter_check_func=_json_missing_essential_fields,
                update_func=update_metadata,
                i=i,
                total_types=total_types,
            )
        elif update_type == UpdateType.PREVIEW_IMAGES.value:
            process_update(
                update_type_name="preview images",
                model_descriptors=model_descriptors,
                wrapped_pr=wrapped_pr,
                overwrite_existing=overwrite_existing,
                filter_check_func=lambda d: not files.preview_exists(d.filename),
                update_func=update_preview_images,
                i=i,
                total_types=total_types,
            )

    # Ensure progress completes
    if wrapped_pr:
        wrapped_pr(fraction=1.0, description="All updates complete")
    time.sleep(1.5)
