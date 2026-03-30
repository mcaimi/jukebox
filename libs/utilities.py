# Utility module
try:
    import os
    import requests
except ImportError as e:
    print(f"{e}")

# set up proxy environment variables
def set_proxy(proxies: dict = None) -> None:
    """
    Sets up proxy environment variables for HTTP and HTTPS
    """
    if not isinstance(proxies, dict):
        raise TypeError("Proxies must be a dictionary")

    # set proxy
    os.environ["HTTP_PROXY"] = proxies.get('http', '')
    os.environ["HTTPS_PROXY"] = proxies.get('https', '')

# download a file with a custom path and optional proxy setting
def download_file(url: str, path: str, proxy: dict = None) -> bool:
    """
    Downloads a file from a URL to a customizable path.

    Args:
        url (str): The URL of the file to be downloaded.
        path (str): The local path where the file will be saved.
        proxy (dict, optional): Proxy servers, in dict format (e.g. {'http': "url", 'https': "url"}). Defaults to None.

    Returns:
        bool: True if the download was successful, False otherwise.
    """

    if proxy is not None:
        if not isinstance(proxy, dict):
            raise ValueError("proxy is not a proper dictionary")

    # download from the net
    try:
        # local path sanity check
        path_fragments = path.split("/")
        if len(path_fragments) > 1:
            # format is path/filename, so we need to create the directory first
            os.makedirs("/".join(path_fragments[:-1]), exist_ok=True)

        # Create the full path where the file will be saved
        local_path = f"{path}"

        # Send a GET request to download the file, following any redirects (including 302)
        response = requests.get(url, stream=True, proxies=proxy, allow_redirects=True)

        # Check if the file was successfully downloaded
        with open(local_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        return True

    except requests.RequestException as e:  # Catch the broader category of requests-related exceptions
        print(f"Error downloading file: {e}")
        return False