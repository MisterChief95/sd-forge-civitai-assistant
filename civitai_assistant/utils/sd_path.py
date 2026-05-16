import os
from collections.abc import Callable
from typing import Optional

from civitai_assistant.const import SAFETENSORS
from civitai_assistant.utils.logger import logger
from civitai_assistant.types import ModelType

from modules.shared import cmd_opts, opts
from modules.paths_internal import models_path, data_path


def get_checkpoint_dirs() -> list[str]:
    """Get all checkpoint directories including default and additional dirs from cmd_opts."""
    dirs = [os.path.join(models_path, "Stable-diffusion")]
    if cmd_opts.ckpt_dirs:
        dirs.extend(cmd_opts.ckpt_dirs)
    return [os.path.abspath(d) for d in dirs]


def get_lora_dirs() -> list[str]:
    """Get all LoRA directories including default and additional dirs from cmd_opts."""
    dirs = [os.path.join(models_path, "Lora")]
    if cmd_opts.lora_dirs:
        dirs.extend(cmd_opts.lora_dirs)
    return [os.path.abspath(d) for d in dirs]


def get_embeddings_dirs() -> list[str]:
    """Get all embeddings directories."""
    dirs = []
    if cmd_opts.embeddings_dir:
        dirs.append(os.path.abspath(cmd_opts.embeddings_dir))
    else:
        dirs.append(os.path.abspath(os.path.join(data_path, "embeddings")))
        dirs.append(os.path.abspath(os.path.join(models_path, "embeddings")))
    return dirs


MODEL_TYPE_TO_DIRECTORY: dict[ModelType, Callable[[], list[str]]] = {
    ModelType.CHECKPOINT: get_checkpoint_dirs,
    ModelType.LORA: get_lora_dirs,
    ModelType.TEXTUAL_INVERSION: get_embeddings_dirs,
}


MAX_SCAN_DEPTH = 4


def find_model_files(model_types: list[ModelType]) -> list[str]:
    """
    Finds all model files of the specified types.
    Args:
        model_types (list[ModelType]): A list of model types to search for.
    Returns:
        list[str]: A list of paths to the model files.
    """

    model_files = []

    for modelType in model_types:
        model_dirs_func = MODEL_TYPE_TO_DIRECTORY.get(modelType)
        if model_dirs_func is None:
            logger.warning(f"Unknown or unselected model type: {modelType}")
            continue

        model_dirs = model_dirs_func()
        for model_dir in model_dirs:
            if not os.path.exists(model_dir):
                logger.debug(f"Model directory does not exist: {model_dir}")
                continue

            max_depth = getattr(opts, "ca_max_scan_depth", MAX_SCAN_DEPTH)
            follow_symlinks = getattr(opts, "ca_follow_symlinks", False)
            base_depth = model_dir.rstrip(os.sep).count(os.sep)
            for root, dirs, files in os.walk(model_dir, followlinks=follow_symlinks):
                current_depth = root.rstrip(os.sep).count(os.sep) - base_depth
                if current_depth >= max_depth:
                    dirs.clear()
                    continue
                for file in files:
                    if file.endswith(SAFETENSORS):
                        model_files.append(os.path.join(root, file))

    logger.debug(f"Found {len(model_files)} models to update")

    return model_files
