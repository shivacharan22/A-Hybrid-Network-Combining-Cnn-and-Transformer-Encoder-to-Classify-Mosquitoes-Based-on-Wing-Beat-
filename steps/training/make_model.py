from typing_extensions import Annotated
import torch
from utils.model import wavmosLit
from zenml import step


@step
def make_model() -> Annotated[torch.nn.Module, "model"]:
    """ 
        Creating the model
        Returns:    
            model: The model
    """
    try:
        model = wavmosLit()
        return model
    except:
        print(" wavmosLit has Failed, please check the class")