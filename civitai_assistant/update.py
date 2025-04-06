import os
import time
import asyncio
from typing import Callable, Any

import gradio as gr
from bs4 import BeautifulSoup as soup

import civitai_assistant.api as api
import civitai_assistant.utils.files as files
import civitai_assistant.utils.sd_path as sd_path
from civitai_assistant.utils.logger import logger
from civitai_assistant.types import CivitaiModel, ModelDescriptor, ModelType

from modules.extra_networks import parse_prompt
from modules.shared import opts


async def process_models_async(
    model_types: list[ModelType],
    overwrite_existing: bool,
    recalculate_hash: bool,
    processor_fn: Callable[[ModelDescriptor, CivitaiModel], None],
    pr: gr.Progress,
    filter_fn: Callable[[str], bool] = None,
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

    # Find model files
    pr(0.1, "Finding model files")
    model_files: list[str] = sd_path.find_model_files(model_types)

    if not model_files:
        logger.info("No models found")
        gr.Info("No models found")
        pr(1.0, "No models found")
        time.sleep(1.5)
        return

    # Filter based on overwrite preference
    pr(0.2, "Checking for overwrite")
    if not overwrite_existing and filter_fn:
        model_files = [file for file in model_files if filter_fn(file)]

    if not model_files:
        logger.info("No models after filtering")
        gr.Info("No models found")
        pr(1.0, "Done")
        time.sleep(1.5)
        return

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

    # Process models in batches for better performance without overwhelming the API
    total_models = len(model_descriptors)
    for i in range(0, total_models, batch_size):
        batch = model_descriptors[i : i + batch_size]

        # Update progress
        progress_start = 0.3 + (i / total_models) * 0.6
        progress_end = 0.3 + ((i + len(batch)) / total_models) * 0.6
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
            for descriptor in batch:
                model_hash = descriptor.metadata_descriptor.hash
                civitai_model = civitai_models.get(model_hash)

                batch_progress = progress_start + (batch.index(descriptor) / len(batch)) * (
                    progress_end - progress_start
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
                    logger.error(f"Error processing {descriptor.file_basename}: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            continue
        
        # Give a small delay between batches to let resources clean up
        await asyncio.sleep(0.5)

    pr(1.0, "Done")
    time.sleep(1.5)


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
    if description and not description.isspace():
        descriptor.metadata_descriptor.description = soup(
            description, "html.parser"
        ).get_text()

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
    model_types: list[ModelType],
    overwrite_existing: bool,
    recalculate_hash: bool,
    pr: gr.Progress = gr.Progress(),  # noqa: B008
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
            model_types=model_types,
            overwrite_existing=overwrite_existing,
            recalculate_hash=recalculate_hash,
            pr=pr,
            processor_fn=metadata_processor_async,
            filter_fn=lambda file: not files.has_json(file),
        )
    )


def update_preview_images(
    model_types: list[ModelType],
    overwrite_existing: bool,
    recalculate_hash: bool,
    pr: gr.Progress = gr.Progress(),  # noqa: B008
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
            model_types=model_types,
            overwrite_existing=overwrite_existing,
            recalculate_hash=recalculate_hash,
            pr=pr,
            processor_fn=image_processor_async,
            filter_fn=lambda file: not files.preview_exists(file),
        )
    )
