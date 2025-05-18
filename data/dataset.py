import os
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image


class FlickrDataset(Dataset):
    def __init__(self, images_folder_path, captions_file_path, vocabulary, transform=None):
        self.images_folder_path = images_folder_path
        self.captions_file_path = captions_file_path

        self.captions_table = pd.read_csv(self.captions_file_path, sep=',',
                                          header=0, names=['image_name', 'caption'],
                                          engine='c', memory_map=True)

        self.vocabulary = vocabulary
        self.transform = transform

    def __getitem__(self, index):
        item = {}

        image_name = self.captions_table.at[index, 'image_name']
        item['image'] = Image.open(os.path.join(self.images_folder_path, image_name)).convert("RGB")

        caption = self.captions_table.at[index, 'caption']
        item['caption_indices'] = self.vocabulary.to_indices(string=caption)

        if self.transform is not None:
            item['image'] = self.transform(item['image'])

        return item

    def __len__(self):
        return len(self.captions_table)
