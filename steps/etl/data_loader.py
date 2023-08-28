from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
from zenml import Step
from zenml.client import client

artifact_store = Client().active_stack.artifact_store

@Step(artifact_store=artifact_store)
def data_loader(
    is_inference: Annotated[bool, "Is this a training or inference job?"]
) -> Tuple[Annotated[pd.DataFrame,"dataset"], Annotated[pd.DataFrame,"labels"]]:
    """ Reads the dataset
        Args:
            is_inference: Is this a training or inference job?
        Returns:    
            dataset: The dataset
            labels: The labels
    """
    # Read the dataset
    try:
        dataset = pd.read_csv(config.DATASET_PATH)
    except FileNotFoundError:
        raise FileNotFoundError("Please download the dataset and place it in the data folder.")

    # Split the dataset into training and inference data
    inference_size = int(len(dataset) * config.INFERENCE_SIZE)
    inference_data = dataset.sample(inference_size, random_state=18)

    if is_inference:
        dataset = inference_data
        dataset.drop(columns=["label"], inplace=True)
    else: 
        dataset.drop(inference_data.index, inplace=True)
        
    dataset.reset_index(drop=True, inplace=True)
    return dataset, inference_data


