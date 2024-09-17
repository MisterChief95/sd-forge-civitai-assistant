from __future__ import annotations
from dataclasses import astuple, dataclass
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, TypeVar, Generic


class MetadataDescriptor(BaseModel):
    hash: Optional[str] = None
    id: Optional[int] = None
    modelId: Optional[int] = None
    description: Optional[str] = None
    sd_version: str = Field(default="Other", alias="sd version")
    activation_text: str = Field(default="", alias="activation text")
    preferred_weight: float = Field(default=0, alias="preferred weight")
    negative_text: str = Field(default="", alias="negative text")
    notes: Optional[str] = ""


class ModelType(str, Enum):
    CHECKPOINT = "Checkpoint"
    LORA = "LoRA/LyCORIS/DoRA"
    TEXTUAL_INVERSION = "Textual Inversion"


class ModelDescriptor:
    metadata_descriptor: MetadataDescriptor = None
    filename: str = None
    requires_write: bool = False
