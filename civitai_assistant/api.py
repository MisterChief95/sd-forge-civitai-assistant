import asyncio
import aiohttp
from typing import Optional, Dict, Any, Union, List, TypeVar
from contextlib import asynccontextmanager

from civitai_assistant.utils.errors import get_exception_msg
from civitai_assistant.utils.logger import logger
from civitai_assistant.types import CivitaiModel


T = TypeVar("T")


# API Endpoints
class CivitaiEndpoints:
    BASE = "https://civitai.com/api/v1"
    MODEL_VERSIONS = f"{BASE}/model-versions"
    BY_HASH = f"{MODEL_VERSIONS}/by-hash/{{hash}}"
    MODELS = f"{BASE}/models"
    MODEL_BY_ID = f"{MODELS}/{{model_id}}"
    DOWNLOAD = "https://civitai.com/api/download/models/{model_version_id}"


class CivitaiAPI:
    """Asynchronous client for Civitai API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Civitai API client.

        Args:
            api_key: Optional API key for authenticated requests
        """
        self.api_key = api_key
        self.session = None
        self.request_timeout = aiohttp.ClientTimeout(total=60)  # 60 seconds timeout

    @asynccontextmanager
    async def get_session(self):
        """Get or create an aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.request_timeout)
            close_session = True
        else:
            close_session = False

        try:
            yield self.session
        finally:
            if close_session:
                await self.session.close()
                self.session = None

    async def _send_request(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send an HTTP request to the Civitai API.

        Args:
            url: The URL to send the request to
            method: HTTP method (default: "GET")
            headers: Optional headers to include
            params: Optional query parameters

        Returns:
            Response data as dictionary or None if the request failed
        """
        if headers is None:
            headers = {}

        if params is None:
            params = {}

        # Add API key if provided
        if self.api_key:
            if method == "GET":
                params["token"] = self.api_key
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with self.get_session() as session:
                async with session.request(
                    method, url, headers=headers, params=params
                ) as response:
                    response.raise_for_status()

                    if response.content_type == "application/json":
                        return await response.json()
                    else:
                        return await response.read()
        except aiohttp.ClientResponseError as e:
            logger.error(f"API request failed with status {e.status}: {e.message}")
            return None
        except Exception as e:
            logger.error(f"API request failed: {get_exception_msg(e)}")
            return None

    async def fetch_by_hash(self, model_hash: str) -> Optional[CivitaiModel]:
        """
        Fetch a model version by its hash.

        Args:
            model_hash: The hash of the model to fetch

        Returns:
            CivitaiModel instance or None if the request failed
        """
        url = CivitaiEndpoints.BY_HASH.format(hash=model_hash)
        response_data = await self._send_request(url)

        if not response_data:
            return None

        try:
            return CivitaiModel.model_validate(response_data)
        except Exception as e:
            logger.error(f"Failed to parse model data: {get_exception_msg(e)}")
            return None

    async def fetch_model_description(self, model_id: Union[str, int]) -> Optional[str]:
        """
        Fetch a model's description by its ID.

        Args:
            model_id: The ID of the model

        Returns:
            Model description as string or None if the request failed
        """
        url = CivitaiEndpoints.MODEL_BY_ID.format(model_id=model_id)
        response_data = await self._send_request(url)

        if not response_data:
            return None

        try:
            return response_data.get("description", "")
        except Exception as e:
            logger.error(f"Failed to extract model description: {get_exception_msg(e)}")
            return None

    async def fetch_image_preview(self, image_url: str) -> Optional[bytes]:
        """
        Fetch an image preview from the given URL.

        Args:
            image_url: The URL of the image to fetch

        Returns:
            Image data as bytes or None if the request failed
        """
        try:
            async with self.get_session() as session:
                async with session.get(image_url) as response:
                    response.raise_for_status()
                    return await response.read()
        except Exception as e:
            logger.error(f"Failed to fetch image preview: {get_exception_msg(e)}")
            return None

    async def fetch_multiple_by_hash(
        self, model_hashes: List[str]
    ) -> Dict[str, Optional[CivitaiModel]]:
        """
        Fetch multiple models by their hashes concurrently.

        Args:
            model_hashes: List of model hashes to fetch

        Returns:
            Dictionary mapping hashes to CivitaiModel instances or None
        """
        tasks = [self.fetch_by_hash(model_hash) for model_hash in model_hashes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            model_hash: result if not isinstance(result, Exception) else None
            for model_hash, result in zip(model_hashes, results)
        }

    async def fetch_multiple_image_previews(
        self, image_urls: List[str]
    ) -> Dict[str, Optional[bytes]]:
        """
        Fetch multiple image previews concurrently.

        Args:
            image_urls: List of image URLs to fetch

        Returns:
            Dictionary mapping URLs to image data or None
        """
        tasks = [self.fetch_image_preview(url) for url in image_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            url: result if not isinstance(result, Exception) else None
            for url, result in zip(image_urls, results)
        }


# Provide backward compatibility functions for synchronous code
_api_instance = None


def get_api_instance(api_key: Optional[str] = None) -> CivitaiAPI:
    """Get or create a global API instance."""
    global _api_instance
    if _api_instance is None:
        _api_instance = CivitaiAPI(api_key)
    return _api_instance


def _run_async(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def fetch_by_hash(model_hash: str) -> Optional[CivitaiModel]:
    """Synchronous wrapper for fetch_by_hash."""
    api = get_api_instance()
    return _run_async(api.fetch_by_hash(model_hash))


def fetch_model_description(model_id: Union[str, int]) -> Optional[str]:
    """Synchronous wrapper for fetch_model_description."""
    api = get_api_instance()
    return _run_async(api.fetch_model_description(model_id))


def fetch_image_preview(url: str) -> Optional[bytes]:
    """Synchronous wrapper for fetch_image_preview."""
    api = get_api_instance()
    return _run_async(api.fetch_image_preview(url))
