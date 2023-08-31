from zenml.config import DockerSettings
from zenml.integrations.constants import (
    PYTORCH,
    PYTORCH_L,
    WANDB
)

# can change if you want
stopping_threshold = 0.95
random_seed = 42
is_inference = False
INFERENCE_SIZE = 0.2
DATASET_PATH = "data/dataset_mos256.csv"
k_folds = 10
MODEL_PATH = "model/model.pt"
# The pipeline settings
PIPELINE_SETTINGS = dict(
    docker=DockerSettings(
        required_integrations=[
            PYTORCH,
            PYTORCH_L,
            WANDB
        ],
    )
)

