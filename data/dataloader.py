import torch

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def get_dataloader(dataset, batch_size=4, shuffle=False, num_workers=1):
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      collate_fn=CaptionsCollate(vocabulary=dataset.vocabulary),
                      pin_memory=True)


class CaptionsCollate(object):
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, items):
        captions = [item['caption_indices'] for item in items]
        padding_value = self.vocabulary.to_index(self.vocabulary.pad_token)

        batch = {'image': torch.stack([item['image'] for item in items], dim=0),
                 'caption_indices': pad_sequence(captions,
                                                 batch_first=True,
                                                 padding_value=padding_value)}

        return batch
