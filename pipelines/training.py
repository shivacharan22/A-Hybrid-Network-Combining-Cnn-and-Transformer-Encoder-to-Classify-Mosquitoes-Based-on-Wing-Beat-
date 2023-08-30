from typing import List, Optional

from steps import (
    data_loader,
    pytorch_dataset_load_split,
    make_model,
    model_trainer,
)
from zenml import Pipeline


@pipeline(
    name="training_pipeline",
    settings=PIPELINE_SETTINGS,
    extra=DEFAULT_PIPELINE_EXTRAS
)
def training_pipeline(random_seed: int) -> None:
    """
        Training pipeline for the mosquito wingbeat dataset.
        Args:
    """
    dataset, inference_data = data_loader()
    pytorch_dataset, spilts = pytorch_dataset_load_split(dataset)
    model = make_model()
    model_trainer(model, random_seed=random_seed)
    None