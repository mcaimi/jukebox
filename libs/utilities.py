# Utility module
try:
    import os
    import requests
    import boto3
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

# download a file from an S3 bucket
def download_from_s3(s3_url: str,
                    s3_bucket: str,
                    remote_object: str,
                    local_object: str,
                    s3_config: dict):
    """
    Downloads a file from an S3 bucket.

    Args:
        s3_url (str): The S3 Endpoint URL.
        s3_bucket (str): The bucket name.
        remote_object (str): The remote object stored in the bucket.
        local_object (str): The local path where the file will be saved.
        s3_config (dict): A dictionary containing AWS credentials for the S3 endpoint. Keys should be 'aws_access_key_id' and 'aws_secret_access_key'.

    Returns:
        bool: True if the download was successful, False otherwise.
    """

    # sanity check: make sure s3_config is indeed of type 'dict'
    if not isinstance(s3_config, dict):
        raise ValueError("s3_config must be of type dict")

    # Set up AWS S3 connection
    session = boto3.Session(aws_access_key_id=s3_config['aws_access_key_id'],
                            aws_secret_access_key=s3_config['aws_secret_access_key'])

    # Create an S3 client object
    s3 = session.client(
        service_name='s3',
        endpoint_url=s3_url,
    )

    # make sure destination path exists
    path_fragments = local_object.split("/")
    if len(path_fragments) > 1:
        # format is path/filename, so we need to create the directory first
        os.makedirs("/".join(path_fragments[:-1]), exist_ok=True)

    # Explicitly specify the S3 endpoint if needed
    try:
        s3.download_file(s3_bucket, remote_object, local_object)
    except Exception as e:  # Catch all exceptions
        print(f"Error downloading file from S3: {e}")
        return False

    # OK, file downloaded
    return True
