from typing import List, Optional
from config import random_seed, is_inference, k_folds


from steps.etl.data_loader import data_loader
from steps.etl.pytorch_dataset_load_split import pytorch_dataset_load_split
from steps.training.make_model import make_model
from steps.training.model_trainer import model_trainer
from config import PIPELINE_SETTINGS
from zenml import pipeline


@pipeline(
    name="training_pipeline",
    settings=PIPELINE_SETTINGS
)
def training_pipeline():
    """
        Training pipeline for the mosquito wingbeat dataset.
        Args:
    """
    dataset, inference_data = data_loader(is_inference)
    pytorch_dataset, spilts = pytorch_dataset_load_split(dataset,k_folds)
    model = make_model()
    model_trainer(model, random_seed=random_seed)
    