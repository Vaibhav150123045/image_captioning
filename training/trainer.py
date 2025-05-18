import torch

from metric import BLEUScore
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, optimizer, checkpointer, logger, device, last_epoch):
        self.model = model
        self.optimizer = optimizer

        ignore_index = self.model.vocabulary.to_index(self.model.vocabulary.pad_token)
        self.objective_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.bleu_score = BLEUScore(weights=[(1.0,), (0.5, 0.5)]).compute_sentence_bleu

        self.checkpointer = checkpointer
        self.logger = logger

        self.device = device

        self.last_epoch = last_epoch

    def train(self, train_dataloader, val_dataloader, num_epochs):
        for epoch in range(self.last_epoch + 1, self.last_epoch + 1 + num_epochs):
            self.run_epoch(dataloader=train_dataloader, epoch=epoch, phase='train')
            self.run_epoch(dataloader=val_dataloader, epoch=epoch, phase='val')

            self.checkpointer.save_checkpoint(epoch=epoch, model=self.model, optimizer=self.optimizer)

    def run_epoch(self, dataloader, epoch, phase):
        is_train = (phase == 'train')

        self.model.train() if is_train else self.model.eval()
        loss = 0.0

        wrapped_iterator = tqdm(iterable=dataloader, desc=f'Epoch {epoch:02} | {phase}', leave=False, unit='batch')
        with torch.set_grad_enabled(is_train):
            for item in wrapped_iterator:
                image = item['image'].to(self.device)
                caption_indices = item['caption_indices'].to(self.device)

                output = self.model(image, caption_indices[:, :-1])
                objective = self.objective_function(output['logits'], caption_indices[:, 1:])
                loss += objective.item()

                wrapped_iterator.set_postfix_str(f"cross_entropy_loss: {objective.item():.4f}")

                if is_train:
                    self._optimize(objective)

            loss = loss / len(dataloader)

        if self.logger is not None:
            self.logger.add_scalars_dict(cross_entropy_loss=loss)

            images = []
            captions = []
            generated_captions = []
            bleu_scores = []
            for i in range(min(4, image.shape[0])):
                images.append(image[i])

                caption_tokens = dataloader.dataset.vocabulary.to_tokens(indices=caption_indices[i],
                                                                         remove_special_tokens=True)
                captions.append(" ".join(caption_tokens))

                generated_caption_tokens = self.model.generate_image_caption_tokens(image=image[i].unsqueeze(dim=0))
                generated_captions.append(" ".join(generated_caption_tokens))

                bleu_scores.append(self.bleu_score(generated_tokens=generated_caption_tokens,
                                                   target_tokens=caption_tokens))

            self.logger.add_table(images=images,
                                  captions=captions,
                                  generated_captions=generated_captions,
                                  bleu_1_scores=[scores[0] for scores in bleu_scores],
                                  bleu_2_scores=[scores[1] for scores in bleu_scores])

            self.logger.log(tag=phase, step=epoch)

    def _optimize(self, objective):
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
