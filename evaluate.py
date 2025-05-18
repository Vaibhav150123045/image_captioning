import os
import argparse
import torch

from utils import get_device, make_reproducible

from data.vocabulary import Vocabulary

from models.utils import get_model_class

from metric import BLEUScore

DEFAULT_VOCABULARY_CAPTIONS_FILE_PATH = os.path.join(".", "flickr8k", "vocabulary_captions.txt")
DEFAULT_VAL_IMAGES_FOLDER_PATH = os.path.join(".", "flickr8k", "val_images")
DEFAULT_VAL_CAPTIONS_FILE_PATH = os.path.join(".", "flickr8k", "val_captions.txt")


class ArgParser(object):
    """Parses cmd args."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(add_help=True)
        self._add_args(parser=self.parser)

    @staticmethod
    def _add_args(parser):
        parser.add_argument('-d', '--device-id', type=int, required=True, help="cuda device id")

        parser.add_argument('--checkpoint-path', type=str, required=True,
                            help="path to the model checkpoint .pth.tar file")
        parser.add_argument('--vocabulary-captions-file-path', default=DEFAULT_VOCABULARY_CAPTIONS_FILE_PATH, type=str,
                            help="path to the captions file used for vocabulary building " +
                                 f"(default: {DEFAULT_VOCABULARY_CAPTIONS_FILE_PATH})")
        parser.add_argument('--val-images-folder-path', default=DEFAULT_VAL_IMAGES_FOLDER_PATH, type=str,
                            help="path to the images folder with val images " +
                                 f"(default: {DEFAULT_VAL_IMAGES_FOLDER_PATH})")
        parser.add_argument('--val-captions-file-path', default=DEFAULT_VAL_CAPTIONS_FILE_PATH, type=str,
                            help="path to the captions file with val captions " +
                                 f"(default: {DEFAULT_VAL_CAPTIONS_FILE_PATH})")

        parser.add_argument('--seed', default=0, type=int, help="seed for reproducibility (default: 0)")

    def parse_args(self):
        """Parses cmd args.

        :return: args namespace (argparse)
        """
        args = self.parser.parse_args()

        return args


def get_model(checkpoint_path, vocabulary, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = get_model_class(model_name=checkpoint['model_config']['name'])(vocabulary=vocabulary,
                                                                           **checkpoint['model_config']['parameters'])
    model.to(device)
    model.load_state_dict(state_dict=checkpoint['model_state_dict'])
    model.eval()

    return model


def save_scores(scores_dict, save_dir):
    with open(os.path.join(save_dir, "bleu_scores.txt"), 'w') as file:
        for score_name, score_value in scores_dict.items():
            file.write(f"{score_name} {score_value}\n")


def main(args):
    make_reproducible(seed=args.seed)
    device = get_device(device_id=args.device_id)

    vocabulary = Vocabulary(captions_file_path=args.vocabulary_captions_file_path)

    model = get_model(checkpoint_path=args.checkpoint_path, vocabulary=vocabulary, device=device)

    bleu_score = BLEUScore(weights=[(1.0,),
                                    (0.5, 0.5),
                                    (0.333, 0.333, 0.334),
                                    (0.25, 0.25, 0.25, 0.25)])
    scores = bleu_score.compute_corpus_bleu(model=model,
                                            images_folder_path=args.val_images_folder_path,
                                            captions_file_path=args.val_captions_file_path,
                                            vocabulary=vocabulary,
                                            device=device)

    save_scores(scores_dict={f"BLEU_{i + 1}": scores[i] for i in range(len(scores))},
                save_dir=os.path.split(args.checkpoint_path)[0])


if __name__ == "__main__":
    arg_parser = ArgParser()
    args = arg_parser.parse_args()

    main(args=args)
