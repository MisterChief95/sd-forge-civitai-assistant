import asyncio
import aiohttp
from typing import Optional, Dict, Any, Union, List, TypeVar

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


# Use request timeout as a constant but don't create a global session
_request_timeout = aiohttp.ClientTimeout(total=10)  # 10 seconds timeout


async def create_session() -> aiohttp.ClientSession:
    """Create a fresh aiohttp session with timeout."""
    return aiohttp.ClientSession(timeout=_request_timeout)


async def _send_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Send an HTTP request to the Civitai API.

    Args:
        url: The URL to send the request to
        method: HTTP method (default: "GET")
        headers: Optional headers to include
        params: Optional query parameters
        api_key: Optional API key for authenticated requests

    Returns:
        Response data as dictionary or None if the request failed
    """
    if headers is None:
        headers = {}

    if params is None:
        params = {}

    # Add API key if provided
    if api_key:
        if method == "GET":
            params["token"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"

    # Create a fresh session for each request
    session = None
    try:
        session = await create_session()
        async with session.request(
            method, url, headers=headers, params=params
        ) as response:
            response.raise_for_status()

            if response.content_type == "application/json":
                return await response.json()
            else:
                return await response.read()

    except aiohttp.ClientError as e:
        logger.error(f"API request failed: {get_exception_msg(e)}")
        return None

    except asyncio.TimeoutError:
        logger.error("API request timed out")
        return None

    except Exception as e:
        logger.error(f"Unexpected error: {get_exception_msg(e)}")
        return None
    finally:
        # Ensure session is closed
        if session and not session.closed:
            await session.close()


async def fetch_by_hash(
    model_hash: str, api_key: Optional[str] = None
) -> Optional[CivitaiModel]:
    """
    Fetch a model version by its hash.

    Args:
        model_hash: The hash of the model to fetch
        api_key: Optional API key for authenticated requests

    Returns:
        CivitaiModel instance or None if the request failed
    """
    url = CivitaiEndpoints.BY_HASH.format(hash=model_hash)
    response_data = await _send_request(url, api_key=api_key)

    if not response_data:
        return None

    try:
        return CivitaiModel.model_validate(response_data)
    except Exception as e:
        logger.error(f"Failed to parse model data: {get_exception_msg(e)}")
        return None


async def fetch_model_description(
    model_id: Union[str, int], api_key: Optional[str] = None
) -> Optional[str]:
    """
    Fetch a model's description by its ID.

    Args:
        model_id: The ID of the model
        api_key: Optional API key for authenticated requests

    Returns:
        Model description as string or None if the request failed
    """
    url = CivitaiEndpoints.MODEL_BY_ID.format(model_id=model_id)
    response_data = await _send_request(url, api_key=api_key)

    if not response_data:
        return None

    try:
        return response_data.get("description", "")
    except Exception as e:
        logger.error(f"Failed to extract model description: {get_exception_msg(e)}")
        return None


async def fetch_image_preview(image_url: str) -> Optional[bytes]:
    """
    Fetch an image preview from the given URL.

    Args:
        image_url: The URL of the image to fetch

    Returns:
        Image data as bytes or None if the request failed
    """
    session = None
    try:
        session = await create_session()
        async with session.get(image_url) as response:
            response.raise_for_status()
            return await response.read()
    except Exception as e:
        logger.error(f"Failed to fetch image preview: {get_exception_msg(e)}")
        return None
    finally:
        # Ensure session is closed
        if session and not session.closed:
            await session.close()


async def fetch_multiple_by_hash(
    model_hashes: List[str], api_key: Optional[str] = None
) -> Dict[str, Optional[CivitaiModel]]:
    """
    Fetch multiple models by their hashes concurrently.

    Args:
        model_hashes: List of model hashes to fetch
        api_key: Optional API key for authenticated requests

    Returns:
        Dictionary mapping hashes to CivitaiModel instances or None
    """
    tasks = [fetch_by_hash(model_hash, api_key) for model_hash in model_hashes]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        model_hash: result if not isinstance(result, Exception) else None
        for model_hash, result in zip(model_hashes, results)
    }


async def fetch_multiple_image_previews(
    image_urls: List[str],
) -> Dict[str, Optional[bytes]]:
    """
    Fetch multiple image previews concurrently.

    Args:
        image_urls: List of image URLs to fetch

    Returns:
        Dictionary mapping URLs to image data or None
    """
    tasks = [fetch_image_preview(url) for url in image_urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        url: result if not isinstance(result, Exception) else None
        for url, result in zip(image_urls, results)
    }
