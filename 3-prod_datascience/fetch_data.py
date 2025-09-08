# Import objects from KubeFlow DSL Library
from kfp.dsl import (
    component,
    Output,
    Dataset,
)


@component(base_image='python:3.11',
           packages_to_install=["dask[dataframe]==2024.8.0",
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

    song_properties = pd.read_parquet('https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_properties.parquet')
    song_rankings = pd.read_parquet('https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_rankings.parquet')

    data = song_rankings.merge(song_properties, on='spotify_id', how='left')
    # remove incomplete data from the dataset
    data.dropna()

    dataset.path += ".parquet"
    dataset.metadata = {
        "song_properties": "https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_properties.parquet",
        "song_rankings": "https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_rankings.parquet"
    }
    data.to_parquet(dataset.path, index=False)


@component(base_image='python:3.11',
           packages_to_install=["feast==0.40.0",
                                "psycopg2>=2.9",
                                "dask-expr==1.1.10",
                                "s3fs==2024.6.1",
                                "psycopg_pool==3.2.3",
                                "psycopg==3.2.3",
                                "pandas==2.2.3",
                                "numpy==1.26.4"])
def fetch_data_from_feast(
    version: str,
    dataset: Output[Dataset]
):
    """
    Fetches data from Feast
    """

    import feast
    import pandas as pd
    import numpy as np

    fs_config_json = {
        'project': 'music',
        'provider': 'local',
        'registry': {
            'registry_type': 'sql',
            'path': 'postgresql://feast:feast@feast:5432/feast',
            'cache_ttl_seconds': 60,
            'sqlalchemy_config_kwargs': {
                'echo': False, 
                'pool_pre_ping': True
            }
        },
        'online_store': {
            'type': 'postgres',
            'host': 'feast',
            'port': 5432,
            'database': 'feast',
            'db_schema': 'feast',
            'user': 'feast',
            'password': 'feast'
        },
        'offline_store': {'type': 'file'},
        'entity_key_serialization_version': 2
    }

    fs_config = feast.repo_config.RepoConfig(**fs_config_json)
    fs = feast.FeatureStore(config=fs_config)

    song_rankings = pd.read_parquet('https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_rankings.parquet')

    # Feast will remove rows with identical id and date so we add a small delta to each
    microsecond_deltas = np.arange(0, len(song_rankings))*2
    song_rankings['snapshot_date'] = pd.to_datetime(song_rankings['snapshot_date'])
    song_rankings['snapshot_date'] = song_rankings['snapshot_date'] + pd.to_timedelta(microsecond_deltas, unit='us')

    feature_service = fs.get_feature_service("serving_fs")

    data = fs.get_historical_features(entity_df=song_rankings, features=feature_service).to_df()

    features = [f.name for f in feature_service.feature_view_projections[0].features]

    dataset.metadata = {
        "song_properties": "serving_fs",
        "song_rankings": "https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_rankings.parquet",
        "features": features
    }
    dataset.path += ".parquet"
    data.to_parquet(dataset.path, index=False)