from typing import Union, List
from pathlib import Path
import numpy as np
import yaml
from monai.transforms import Activations
import lib.utils as utils

class TopoMortar:

    classes = ["mortar"]
    dim = 2

    def __init__(self, labels: str, training_set_size: str) -> None:
        self.labels = f"{labels}"
        self.training_set_size = training_set_size
        self.save_predictions_fn = utils.save_predictions_PNG_binary

    def split(self, path_split: Path, fold: int) -> Union[List, List, List]:

        with open(path_split, 'r') as f:
            split = yaml.load(f, Loader=yaml.SafeLoader)[fold]

        train, val, test = [], [], []

        for t in split['train']:
            label_path = Path(t.replace("images", self.labels))
            train.append({'image': t, 'label': label_path, 'id': Path(t).stem})
        for t in split['val']:
            label_path = Path(t.replace("images", self.labels))
            val.append({'image': t, 'label': label_path, 'id': Path(t).stem})
        for t in split['test']:
            label_path = Path(t.replace("images", "accurate"))
            test.append({'image': t, 'label': str(label_path), 'id': Path(t).stem})

        return train, val, test
