import logging
import numpy as np
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, size: int = None, data_transforms = None):
        self.images_dir = Path(images_dir)
        self.data_transforms = data_transforms
        self.ids = [splitext(file)[0] for file in listdir(images_dir)
                    if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        if size:
            self.ids = self.ids[:size]

        logging.info(f'Creating dataset with {len(self.ids)} examples')


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = list(self.images_dir.glob(name + '.*'))


        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        img = load_image(img_file[0])
        if self.data_transforms:
            img = self.data_transforms(img)
        return img
