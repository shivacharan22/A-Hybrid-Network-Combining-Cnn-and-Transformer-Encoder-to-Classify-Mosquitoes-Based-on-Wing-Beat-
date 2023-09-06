from typing import Union

import torchaudio
from torch import Tensor
import pandas as pd


def transform_inpu(input : Union[str, pd.DataFrame]) -> Tensor:
    """
    Transform the input audio data to have a sample rate of 8000Hz for use in PyTorch datasets.

    Args:
        input (Union[str, pd.DataFrame]): The path to the audio file or a pandas DataFrame containing audio data.

    Returns:
        Tensor: Transformed audio data with a sample rate of 8000Hz.

    Example:
        waveform = transform_inpu('path/to/audio.wav')  # Transform audio file.
        transformed_data = transform_inpu(data_frame)  # Transform audio data in a DataFrame.
    """
    waveform2, sample_rate2 = torchaudio.load(input)
    if sample_rate2 != 8000:
        waveform2 = torchaudio.functional.resample(waveform2, sample_rate2,8000)
        return waveform2[0] 
