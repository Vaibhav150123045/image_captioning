import spacy
import pandas as pd
import torch

from tqdm import tqdm


class Vocabulary(object):
    """Builds a vocabulary from a file with captions.

    A vocabulary is a mapping between indices and tokens (words, punctuation marks, etc.):

    0 <--> "<PAD>" (padding token)
    1 <--> "<SOS>" (start of sentence token)
    2 <--> "<EOS>" (end of sentence token)
    3 <--> "<UNK>" (unknown token)
    4 <--> ...

    Args:
        captions_file_path: path to the file with captions.
        frequency_threshold: tokens occurring less than frequency_threshold times will be considered as "<UKN>".
    """
    def __init__(self, captions_file_path, frequency_threshold=5):
        self.captions_file_path = captions_file_path
        self.frequency_threshold = frequency_threshold

        self._english_pipeline = spacy.load("en_core_web_sm")

        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unknown_token = "<UNK>"

        self._itos = {0: self.pad_token, 1: self.sos_token, 2: self.eos_token, 3: self.unknown_token}  # index to string
        self._stoi = {self.pad_token: 0, self.sos_token: 1, self.eos_token: 2, self.unknown_token: 3}  # string to index

        self._build_vocabulary()

    def __len__(self):
        return len(self._itos)

    def _tokenize(self, string):
        index = string.rfind(".")
        if index != -1:
            string = string[:index - 1]

        return [token.text.lower() for token in self._english_pipeline.tokenizer(string)]

    def _build_vocabulary(self):
        captions = pd.read_csv(self.captions_file_path, sep=',', engine='c', memory_map=True)["caption"].tolist()

        frequencies = {}
        index = len(self._stoi)

        for string in tqdm(iterable=captions, desc=f'Building vocabulary', leave=False, unit='caption'):
            for token in self._tokenize(string):
                frequencies[token] = frequencies.get(token, 0) + 1

                if frequencies[token] == self.frequency_threshold:
                    self._stoi[token] = index
                    self._itos[index] = token
                    index += 1

    def to_index(self, token):
        return self._stoi.get(token, self._stoi[self.unknown_token])

    def to_token(self, index):
        return self._itos[index]

    def to_indices(self, string):
        tokenized_string = self._tokenize(string)
        indices = [self.to_index(token) for token in tokenized_string]

        indices = [self.to_index(self.sos_token),] + indices + [self.to_index(self.eos_token),]

        return torch.tensor(indices).to(torch.int64)

    def to_tokens(self, indices, remove_special_tokens=True):
        if isinstance(indices, torch.Tensor):
            indices_list = indices.tolist()
        else:
            indices_list = indices

        if remove_special_tokens:
            self._remove_special_tokens(indices_list)

        return [self.to_token(index) for index in indices_list]

    def _remove_special_tokens(self, indices_list):
        pad_token_index = self.to_index(self.pad_token)
        if pad_token_index in indices_list:
            del indices_list[indices_list.index(pad_token_index):]

        sos_token_index = self.to_index(self.sos_token)
        if sos_token_index in indices_list:
            del indices_list[indices_list.index(sos_token_index)]

        eos_token_index = self.to_index(self.eos_token)
        if eos_token_index in indices_list:
            del indices_list[indices_list.index(eos_token_index)]
