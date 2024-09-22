import os
from collections.abc import Callable
from typing import Optional

from civitai_assistant.const import SAFETENSORS
from civitai_assistant.utils.logger import logger
from civitai_assistant.type import ModelType

from modules.shared import cmd_opts
from modules.sd_models import model_path


MODEL_TYPE_TO_DIRECTORY: dict[ModelType, Callable[[], str]] = {
    ModelType.CHECKPOINT: lambda: os.path.abspath(cmd_opts.ckpt_dir or model_path),
    ModelType.LORA: lambda: os.path.abspath(cmd_opts.lora_dir),
    ModelType.TEXTUAL_INVERSION: lambda: os.path.abspath(cmd_opts.embeddings_dir),
}


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
        model_dir: Optional[str] = MODEL_TYPE_TO_DIRECTORY.get(modelType, lambda: None)()
        if model_dir is None:
            logger.warning(f"Unknown or unselected model type: {modelType}")
            continue

        for root, _, files in os.walk(model_dir):
            for file in files:
                if file.endswith(SAFETENSORS):
                    model_files.append(os.path.join(root, file))

    logger.debug(f"Found {len(model_files)} models to update")

    return model_files
