# Import objects from KubeFlow DSL Library
from kfp.dsl import (
    component,
    Output,
    Dataset,
)


@component(base_image='python:3.11',
           packages_to_install=["dask[dataframe]==2024.8.0",
                                "requests",
                                "s3fs==2025.2.0",
                                "pandas==2.2.3"])
def fetch_data(
    version: str,
    dataset: Output[Dataset]
):
    """
    Fetches data from URL
    """

    import pandas as pd
    import os
    import requests
    
    # get proxy settings from env
    proxy_settings: dict = {
        "http": os.environ.get('HTTP_PROXY', ""),
        "https": os.environ.get('HTTPS_PROXY', "")
    }

    # download from the net
    try:
        # data files to download
        SONG_PROPERTIES: dict = {
            "file": "song_properties.parquet",
            "url": 'https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_properties.parquet',
        }

        SONG_RANKINGS: dict = {
            "file": "song_rankings.parquet",
            "url": 'https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_rankings.parquet',
        }

        for feature_file in [SONG_PROPERTIES, SONG_RANKINGS]:
            # Send a GET request to download the file, following any redirects (including 302)
            response = requests.get(feature_file.get("url"), stream=True, proxies=proxy_settings, allow_redirects=True)

            # Check if the file was successfully downloaded
            with open(feature_file.get("file"), 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)

    except requests.RequestException as e:  # Catch the broader category of requests-related exceptions
        print(f"Error downloading file: {e}")

    song_properties = pd.read_parquet(SONG_PROPERTIES["file"])
    song_rankings = pd.read_parquet(SONG_RANKINGS["file"])

    data = song_rankings.merge(song_properties, on='spotify_id', how='left')
    # remove incomplete data from the dataset
    data.dropna()

    dataset.path += ".parquet"
    dataset.metadata = {
        "song_properties": "https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_properties.parquet",
        "song_rankings": "https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_rankings.parquet"
    }
    data.to_parquet(dataset.path, index=False)