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

    This pipeline includes the following steps:
    1. Load and prepare the dataset for training and inference.
    2. Split the dataset into K folds for cross-validation.
    3. Create and initialize the model.
    4. Train the model using the training data.
    
    Example:
        training_pipeline()

    Args:
        None
    """
    dataset, inference_data = data_loader(is_inference)
    pytorch_dataset, spilts = pytorch_dataset_load_split(dataset,k_folds)
    model = make_model()
    model_trainer(model, random_seed=random_seed)
    
