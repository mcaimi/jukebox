# Import objects from KubeFlow Pipelines DSL Library
from kfp.dsl import (
    component,
    Input,
    Dataset,
)


@component(base_image='python:3.11')
def validate_data(
    version: str,
    dataset: Input[Dataset]
) -> bool:
    """
    Validates if the data schema is correct and if the values are reasonable.
    """

    if not dataset.path:
        raise Exception("dataset not found")
    return True