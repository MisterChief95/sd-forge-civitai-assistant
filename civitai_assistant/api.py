import requests
from typing import Optional
from urllib.parse import urlparse, urlencode

from civitai_assistant.utils.errors import get_exception_msg
from civitai_assistant.utils.logger import logger
from civitai_assistant.type import CivitaiModel


API_BY_HASH = "https://civitai.com/api/v1/model-versions/by-hash/{}"
API_BY_MODEL_ID = "https://civitai.com/api/v1/models/{}"


def fetch_by_hash(model_hash: str) -> Optional[CivitaiModel]:
    """
    Fetches a Civitai model using its hash.
    This function sends a request to the Civitai API to retrieve a model
    based on the provided hash. If the request is successful and the response
    can be validated against the CivitaiModel schema, the function returns
    the CivitaiModel instance. If the request fails or the response cannot
    be validated, the function returns None.
    Args:
        model_hash (str): The hash of the model to fetch.
    Returns:
        Optional[CivitaiModel]: The CivitaiModel instance if the request and
        validation are successful, otherwise None.
    """

    try:
        response = send_request(API_BY_HASH.format(model_hash))

        if not response:
            return None

        civitai_model = CivitaiModel.model_validate(response.json())

        return civitai_model

    except Exception as e:
        logger.error(f"Failed to convert response to CivitaiModel: {str(e)}")

        return None


def fetch_model_description(model_id: str | int) -> Optional[str]:
    """
    Fetches a Civitai model by its ID.
    Args:
        id (str | int): The ID of the model group to fetch. Can be a string or an integer.
    Returns:
        Optional[CivitaiModel]: The fetched Civitai model if successful, otherwise None.
    Raises:
        Exception: If there is an error converting the response to a CivitaiModel.
    """

    try:
        response = send_request(API_BY_MODEL_ID.format(model_id))

        if not response:
            return None

        json = response.json()

        if json and json["description"]:
            return str(json["description"])
        return None

    except Exception as e:
        logger.error(f"Failed to read description for CivitAI model: {str(e)}")

        return None


def fetch_image_preview(url: str) -> Optional[bytes]:
    """
    Fetches the image preview from the given URL.
    This function sends a request to the specified URL and attempts to retrieve
    the image content in bytes. If the request is successful and the content is
    in bytes, it returns the image content. Otherwise, it raises an exception.
    Args:
        url (str): The URL of the image to fetch.
    Returns:
        Optional[bytes]: The image content in bytes if successful, otherwise None.
    """

    try:
        response = send_request(urlparse(url).geturl(), stream=True)

        if response and isinstance(response.content, bytes):
            return response.content

        return None

    except Exception as e:
        logger.error(f"Failed to fetch image preview from {url}: {get_exception_msg(e)}")

        return None


def send_request(
    url: str,
    method: str = "GET",
    api_token: Optional[str] = None,
    headers: Optional[dict] = None,
    stream: Optional[bool] = False,
) -> requests.Response:
    """
    Sends an HTTP request to the specified URL using the provided method and headers.
    Args:
        url (str): The URL to send the request to.
        method (str): The HTTP method to use for the request (default is "GET").
        api_token (str, optional): An optional API token to include in the request.
        headers (dict, optional): Optional headers to include in the request.
        stream (bool, optional): Whether to stream the response content (default is False).
    Returns:
        requests.Response: Response from the API.
    """
    parsed_url = urlparse(url).geturl()

    if api_token:
        parsed_url = parsed_url + urlencode({"api_token": api_token})

    response = requests.request(method, parsed_url, headers=headers, stream=stream)
    response.raise_for_status()

    return response
