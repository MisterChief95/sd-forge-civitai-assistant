import requests

from urllib.parse import urlparse, urlencode
from requests.exceptions import RequestException

from civitai_assistant.logger import logger
from civitai_assistant.type import CivitaiModel


# Ensure valid URL
API_BY_HASH = "https://civitai.com/api/v1/model-versions/by-hash/{}"


def fetch_model_data(model_hash: str) -> CivitaiModel:
    """
    Fetches model data from the Civitai API using the provided model hash.
    Args:
        model_hash (str): The hash of the model to fetch data for.
    Returns:
        Result[CivitaiModel]: A Result object containing either the fetched CivitaiModel data or an error.
    """

    response = send_request(API_BY_HASH.format(model_hash))

    if not response:
        return None

    try:
        CivitaiModel.model_validate(response.json())
        civitai_model = CivitaiModel.model_validate(response.json())

        return civitai_model

    except Exception as e:
        logger.error(f"Failed to convert response to CivitaiModel: {str(e)}")

        return None


def fetch_image_preview(url: str) -> bytes:
    """
    Sends an HTTP request to the specified URL using the provided method and headers.
    Args:
        url (str): The URL to send the request to.
        method (str): The HTTP method to use for the request (default is "GET").
        api_token (str, optional): An optional API token to include in the request.
        headers (dict, optional): Optional headers to include in the request.
        stream (bool, optional): Whether to stream the response content (default is False).
    Returns:
        Result[requests.Response]: A Result object containing either the response or an error.
    """
    response = send_request(urlparse(url).geturl(), stream=True)

    return response.content if response else None


def send_request(
    url: str, method: str = "GET", api_token: str = None, headers: dict = None, stream: bool = False
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
        Result[requests.Response]: A Result object containing either the response or an error.
    """
    parsed_url = urlparse(url).geturl()

    if api_token:
        parsed_url = parsed_url + urlencode({"api_token": api_token})

    try:
        response = requests.request(method, parsed_url, headers=headers, stream=stream)
        response.raise_for_status()

        return response

    except RequestException as e:
        logger.error(f"Failed to send request to {url}: {str(e)}")

        return None
