from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


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

    def __hash__(self):
        return hash(
            (
                self.hash,
                self.id,
                self.modelId,
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
    metadata_descriptor: MetadataDescriptor = None
    filename: str = None

    def __hash__(self):
        return hash((self.metadata_descriptor, self.filename))
