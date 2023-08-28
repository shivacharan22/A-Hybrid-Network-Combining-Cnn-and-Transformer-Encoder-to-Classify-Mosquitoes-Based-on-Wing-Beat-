from typing import Union

import torchaudio
from torch import Tensor
import pandas as pd


def transform_inpu(input : Union[str, pd.DataFrame]) -> Tensor:
    """ transform input to 8000Hz used by pytorch dataset 
        Args:
            input: the path of the audio file or the dataframe
            Returns: transformed Tensor  
    """
    waveform2, sample_rate2 = torchaudio.load(input)
    if sample_rate2 != 8000:
        waveform2 = torchaudio.functional.resample(waveform2, sample_rate2,8000)
        return waveform2[0] 