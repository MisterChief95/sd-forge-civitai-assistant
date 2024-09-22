import pytest

import requests
from typing import Any, Optional

import civitai_assistant.api as api
from civitai_assistant.type import CivitaiModel


class MockResponse:
    def __init__(self, status_code: int, json: dict[Any, Any] = None, content: bytes | Any = None) -> None:
        self.status_code: int = status_code
        self.resp_json: dict[Any, Any] = json
        self.content: bytes | Any = content

    def json(self):
        return self.resp_json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError


class Validator:

    @staticmethod
    def assert_civitai_model(civitai_model: CivitaiModel, status: int) -> None:
        if status >= 400:
            assert civitai_model is None, f"civitai_model is not None for status {status}"
            return

        assert isinstance(civitai_model, CivitaiModel), "received obj that is not CivitaiModel"
        assert civitai_model.id == 1234, "id is not 1234"
        assert civitai_model.modelId == 5678, "modelId is not 5678"
        assert civitai_model.images is not None, "images cannot be None"


@pytest.fixture
def validator():
    return Validator


@pytest.mark.parametrize(
    "status,json",
    [
        (200, {"id": "1234", "modelId": "5678", "images": [{"url": "url", "nsfwLevel": 1, "hasMeta": True}]}),
        (200, {"id": "1234", "modelId": "5678", "baseModel": "SDXL1.0", "images": [], "description": "test"}),
        (500, None),
    ],
)
def test_fetch_model_by_hash(status, json, monkeypatch, validator):
    def mock_request(url: str, *args, **kwargs):
        return MockResponse(status, json=json)

    monkeypatch.setattr(requests, "request", mock_request)
    civitai_model: Optional[CivitaiModel] = api.fetch_model_by_hash("abcd1234")
    validator.assert_civitai_model(civitai_model, status)


@pytest.mark.parametrize(
    "status,json",
    [
        (200, {"id": "1234", "modelId": "5678", "images": [{"url": "url", "nsfwLevel": 1, "hasMeta": True}]}),
        (200, {"id": "1234", "modelId": "5678", "baseModel": "SDXL1.0", "images": [], "description": "test"}),
        (500, None),
    ],
)
def test_fetch_model_by_id(status, json, monkeypatch, validator):
    def mock_request(url: str, *args, **kwargs):
        return MockResponse(status, json=json)

    monkeypatch.setattr(requests, "request", mock_request)
    civitai_model: Optional[CivitaiModel] = api.fetch_model_by_id("1234")
    validator.assert_civitai_model(civitai_model, status)


@pytest.mark.parametrize(
    "status,content",
    [
        (200, b"image content"),
        (404, None),
    ],
)
def test_fetch_image_preview(status, content, monkeypatch):
    def mock_request(url: str, *args, **kwargs):
        return MockResponse(status, content=content)

    monkeypatch.setattr(requests, "request", mock_request)

    image_content: Optional[bytes] = api.fetch_image_preview("http://example.com/image.png")

    if status >= 400:
        assert image_content is None, f"image_content is not None for status {status}"
        return

    assert isinstance(image_content, bytes), "received content that is not bytes"
    assert image_content == b"image content", "image content does not match"
