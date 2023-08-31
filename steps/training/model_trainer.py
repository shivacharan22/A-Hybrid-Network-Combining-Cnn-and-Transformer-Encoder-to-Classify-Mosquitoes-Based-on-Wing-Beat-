from typing_extensions import Annotated
import torch
import pandas as pd
from zenml.integrations.wandb.experiment_trackers import WandbExperimentTracker
from zenml import step
from zenml.client import Client
import pytorch_lightning as pl

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, WandbExperimentTracker
):
    raise RuntimeError(
        "active stack doesnt have a Wandb experiment tracker configured. "
    )

@step(experiment_tracker=experiment_tracker.name)
def model_trainer(
    model: Annotated[torch.nn.Module, "model"],
    random_seed: Annotated[int, "random_seed"],
):
    """ 
        Training the model
        Args:
            dataset: The dataset
            model: The model
            random_seed: The random seed
    """
    # Setting the seed
    #pl.utilities.seed.seed_everything(seed=random_seed, workers=True)
    try:
        trainer = Trainer(
            gpus = 1,log_every_n_steps=1,max_epochs = 80)#,resume_from_checkpoint ='/notebooks/lightning_logs/version_110/checkpoints/epoch=74-step=20775.ckpt')
        trainer.fit(model)
    except:
        print(" Trainer has Failed, please check")
    try:
        # Saving the model
        torch.save(model.state_dict(), config.MODEL_PATH)
        print("Model saved successfully")
    except:
        print("Error!Model not saved")
    
    


