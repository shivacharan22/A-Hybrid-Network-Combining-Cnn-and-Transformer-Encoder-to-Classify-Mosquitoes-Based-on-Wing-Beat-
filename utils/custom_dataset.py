from torch.utils.data import Dataset
import pandas as pd
from utils.preprocess import transform_inpu


class CustomImageDataset(Dataset):

    """ Custom dataset Class for the mosquito wingbeat dataset"""

    def __init__(self, annotations_file):
        self.data = annotations_file
        self.lab_dict = {}
        for index,value in enumerate(self.data["labels"].unique()):
            self.lab_dict[value] = index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data.iloc[idx, 4]
        input = '/storage/'+ '/'.join(input.split('/')[5:])
        transform_input = transform_inpu(input)
        label = self.data.iloc[idx, 5]
        transform_label = self.lab_dict[label]
        return transform_input, transform_label