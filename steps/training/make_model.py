from typing_extensions import Annotated

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
    except:
        print(" wavmosLit has Failed, please check the class")
    return model