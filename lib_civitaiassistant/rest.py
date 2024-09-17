import requests

from urllib.parse import urlparse, urlencode
from requests.exceptions import RequestException

from .types import CivitaiModel, Result


# Ensure valid URL
API_BY_HASH = "https://civitai.com/api/v1/model-versions/by-hash/{}"


def fetch_model_data(model_hash: str) -> Result[CivitaiModel]:
    """
    Fetches model data from the Civitai API using the provided model hash.
    Args:
        model_hash (str): The hash of the model to fetch data for.
    Returns:
        Result[CivitaiModel]: A Result object containing either the fetched CivitaiModel data or an error.
    """
    
    response, error = send_request(API_BY_HASH.format(model_hash), "GET")

    if error:
        return Result(error=error)
    
    try:
        CivitaiModel.model_validate(response.json())
        return Result(value=CivitaiModel.model_validate(response.json()))
    
    except Exception as e:
        return Result(error=e)


def send_request(url: str, method: str = "GET", api_token: str=None, headers: dict=None, stream: bool = False) -> Result[requests.Response]:
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
        response = requests.request(method, parsed_url, headers=headers)
        response.raise_for_status()
        return Result(value=response)
    
    except RequestException as e:
        return Result(error=e)
