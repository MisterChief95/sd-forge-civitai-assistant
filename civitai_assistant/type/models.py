from __future__ import annotations

import os
from enum import Enum
from functools import cached_property
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class MetadataDescriptor(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    hash: str
    id: Optional[int] = None
    model_id: Optional[int] = Field(default=None, alias="model id")
    description: Optional[str] = None
    sd_version: Optional[str] = Field(default="Other", alias="sd version")
    activation_text: Optional[str] = Field(default="", alias="activation text")
    preferred_weight: Optional[float] = Field(default=0, alias="preferred weight")
    negative_text: Optional[str] = Field(default="", alias="negative text")
    notes: Optional[str] = ""

    def __hash__(self):
        return hash(
            (
                self.hash,
                self.id,
                self.model_id,
                self.description,
                self.sd_version,
                self.activation_text,
                self.preferred_weight,
                self.negative_text,
                self.notes,
            )
        )


class ModelType(str, Enum):
    CHECKPOINT = "Checkpoint"
    LORA = "LoRA/LyCORIS/DoRA"
    TEXTUAL_INVERSION = "Textual Inversion"


class ModelDescriptor(BaseModel):
    metadata_descriptor: MetadataDescriptor
    filename: str

    def __hash__(self):
        return hash((self.metadata_descriptor, self.filename))

    @cached_property
    def file_basename(self) -> str:
        return os.path.basename(self.filename)
