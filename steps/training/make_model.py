from typing_extensions import Annotated
import torch
from utils.model import wavmosLit
from zenml import step


@step
def make_model() -> Annotated[torch.nn.Module, "model"]:
    """
    Create and initialize the model.

    Returns:
        torch.nn.Module: The initialized model.

    Example:
        model = make_model()
    """
    try:
        model = wavmosLit()
        return model
    except:
        print(" wavmosLit has Failed, please check the class")
