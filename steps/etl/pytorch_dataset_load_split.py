
from typing import Tuple, Dict
from typing_extensions import Annotated
from torch.utils.data import Dataset
import pandas as pd
from zenml import step
from utils.custom_dataset import CustomImageDataset
from sklearn.model_selection import StratifiedKFold

@step
def pytorch_dataset_load_split(
    dataset: Annotated[pd.DataFrame,"dataset"],
    K: Annotated[int,"k_folds"],
) -> Tuple[ Annotated[Dataset,"pytorch_dataset"], Dict[int, Annotated[pd.DataFrame,"fold_data"]]]:
     """
    Split the dataset into K folds and create a PyTorch dataset.

    Args:
        dataset (pd.DataFrame): The dataset containing inputs and labels.
        K (int): The number of folds for cross-validation.

    Returns:
        Tuple[Dataset, Dict[int, pd.DataFrame]]: A tuple containing the PyTorch dataset and a dictionary
        with fold data.

    Example:
        pytorch_dataset, fold_data = pytorch_dataset_load_split(dataset, K=5)
    """
    # Creating the pytorch dataset
    try:
        Custom_dataset = CustomImageDataset(dataset)
    except:
        print(" CustomImageDataset has Failed, please check the dataset path")

    # Split the dataset into K folds
    spilts = {}
    kfold = StratifiedKFold(n_splits=K, shuffle=True,random_state = 42)
    for fold, data in enumerate(kfold.split(dataset['inputs'],dataset['labels'])):
        spilts[fold] = data

    return Custom_dataset, spilts    
