import os
from tempfile import NamedTemporaryFile

import civitai_assistant.utils.files as files
from civitai_assistant.type import ModelDescriptor
from civitai_assistant.const import SAFETENSORS, JSON


def test_generate_model_descriptor():

    file = NamedTemporaryFile(delete=False, suffix=SAFETENSORS)

    model_descriptor: ModelDescriptor = files.generate_model_descriptor(file.name)

    assert files.has_json(file.name)
    assert model_descriptor.metadata_descriptor is not None
    assert model_descriptor.metadata_descriptor.hash is not None

    os.unlink(file.name.replace(SAFETENSORS, JSON))
    file.close()
