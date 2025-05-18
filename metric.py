import os
import pandas as pd

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from PIL import Image
from tqdm import tqdm

from data.transforms import get_val_transform


class BLEUScore(object):
    def __init__(self, weights=(0.25, 0.25, 0.25, 0.25)):
        self.weights = weights

        self._sentence_bleu = lambda references, hypothesis: sentence_bleu(references=references,
                                                                           hypothesis=hypothesis,
                                                                           weights=self.weights)

        self._corpus_bleu = lambda list_of_references, hypotheses: corpus_bleu(list_of_references=list_of_references,
                                                                               hypotheses=hypotheses,
                                                                               weights=self.weights)

    def compute_sentence_bleu(self, generated_tokens, target_tokens):
        return self._sentence_bleu(references=[target_tokens], hypothesis=generated_tokens)

    def compute_corpus_bleu(self, model, images_folder_path, captions_file_path, vocabulary, device):
        references_table = self._get_references_table(captions_file_path=captions_file_path, vocabulary=vocabulary)

        image_transform = get_val_transform()

        list_of_references = []
        hypotheses = []
        for row in tqdm(iterable=references_table.itertuples(), desc=f'Making hypotheses', total=len(references_table),
                        leave=False, unit='hypothesis'):
            list_of_references.append(row.caption)

            image = image_transform(Image.open(os.path.join(images_folder_path, row.image_name)).convert("RGB"))
            hypotheses.append(model.generate_image_caption_tokens(image=image.unsqueeze(dim=0).to(device)))

        return self._corpus_bleu(list_of_references=list_of_references, hypotheses=hypotheses)

    def _get_references_table(self, captions_file_path, vocabulary):
        captions_table = pd.read_csv(captions_file_path, sep=',',
                                     header=0, names=['image_name', 'caption'],
                                     engine='c', memory_map=True)

        references_table = captions_table.groupby(['image_name'])['caption'].apply(list).reset_index()
        references_table['caption'] = references_table['caption'].apply(self._to_tokens_list_function(vocabulary))

        return references_table

    @staticmethod
    def _to_tokens_list_function(vocabulary):
        return lambda strings: [vocabulary.to_tokens(indices=vocabulary.to_indices(string), remove_special_tokens=True)
                                for string in strings]
