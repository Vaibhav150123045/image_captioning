import os
import wandb


class WandBLogger(object):
    """Logs metrics and media with the interface provided by wandb.ai.

    Args:
        log_dir (str): path to the folder with logs.
        config (dict): configs from config file.
        resume (bool): are the training process resuming or not.
    """
    def __init__(self, log_dir, config=None, resume=False):
        self.log_dir = log_dir

        if resume:
            run_id = self._get_latest_run_id()
        else:
            run_id = wandb.util.generate_id()

        wandb.init(project=os.path.split(os.getcwd())[1],
                   config=config,
                   name=os.path.split(log_dir)[1],
                   dir=log_dir,
                   resume='allow',
                   id=run_id)

        self.scalars_dict = {}
        self.table = None

    def _get_latest_run_id(self):
        latest_run_file_name = [file_name for file_name in os.listdir(os.path.join(self.log_dir, 'wandb', 'latest-run'))
                                if file_name.endswith(".wandb")][0]
        return os.path.splitext(latest_run_file_name)[0].split('-')[-1]

    def log(self, tag, step):
        wandb.log({**self._flatten_dict(dict_={tag: self.scalars_dict}, separator='/'),
                   **self._flatten_dict(dict_={tag: self.table}, separator='.')},
                  step=step)

        self.scalars_dict = {}
        self.table = None

    def add_scalars_dict(self, **scalars_dict):
        self.scalars_dict.update(scalars_dict)

    def add_table(self, images, captions, generated_captions, bleu_1_scores, bleu_2_scores):
        columns = ["image", "caption", "generated caption", "BLEU_1", "BLEU_2"]
        data = [[wandb.Image(image), caption, generated_caption, bleu_1_score, bleu_2_score]
                for image, caption, generated_caption, bleu_1_score, bleu_2_score in zip(images,
                                                                                         captions,
                                                                                         generated_captions,
                                                                                         bleu_1_scores, bleu_2_scores)]

        self.table = wandb.Table(columns=columns, data=data)

    @staticmethod
    def _flatten_dict(dict_, separator='/'):
        prefix = ''
        stack = [(dict_, prefix)]
        flat_dict = {}

        while stack:
            current_dict, current_prefix = stack.pop()
            for key, value in current_dict.items():
                new_key = current_prefix + separator + key if current_prefix else key
                if isinstance(value, dict):
                    stack.append((value, new_key))
                else:
                    flat_dict[new_key] = value

        return flat_dict

    def __del__(self):
        wandb.finish()
