import torch


class BaseModel(torch.nn.Module):
    """Base class for all models."""
    def __init__(self, vocabulary):
        super().__init__()

        self.image_encoder = None
        self.caption_generator = None

        self.vocabulary = vocabulary

    def freeze(self):
        """Calls freeze method from image_encoder and caption_generator.

        The freeze method sets the requires_grad parameter to False for some model parameters.
        """
        self.image_encoder.freeze()
        self.caption_generator.freeze()

    def get_optimizer(self, lr):
        return torch.optim.Adam([parameter for parameter in self.parameters() if parameter.requires_grad], lr=lr)

    def forward(self, image, caption_indices):
        """Forward method.

        :param image: torch.tensor of the shape [batch_size, channels, height, width]
        :param caption_indices: torch.tensor of the shape [batch_size, sequence_length]

        :return: output dict at least with 'logits' and 'indices' keys,
            where: logits is the torch.tensor of the shape [batch_size, vocabulary_size, sequence_length]
                   indices is the torch.tensor of the shape [batch_size, sequence_length]
        """
        encoded_image = self.image_encoder.forward(image=image)
        output = self.caption_generator.forward(encoded_image=encoded_image, caption_indices=caption_indices)

        return output

    def generate_image_caption_tokens(self, image, max_length=50):
        """Generates a caption for an image.

        :param image: torch.tensor of the shape [1, channels, height, width]
        :param max_length: maximum caption length (int)

        :return: caption_tokens (list of tokens)
        """
        self.image_encoder.eval()
        self.caption_generator.eval()

        eos_token_index = self.vocabulary.to_index(self.vocabulary.eos_token)
        sos_token_index = self.vocabulary.to_index(self.vocabulary.sos_token)
        with torch.no_grad():
            encoded_image = self.image_encoder.forward(image=image)
            caption_indices = self.caption_generator.generate_caption_indices(encoded_image=encoded_image,
                                                                              sos_token_index=sos_token_index,
                                                                              eos_token_index=eos_token_index,
                                                                              max_length=max_length)

        caption_tokens = self.vocabulary.to_tokens(indices=caption_indices, remove_special_tokens=True)

        return caption_tokens


class BaseImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def freeze(self):
        """Sets the requires_grad parameter to False for some model parameters."""
        raise NotImplementedError

    def forward(self, image):
        """Forward method.

        :param image: torch.tensor of the shape [batch_size, channels, height, width]

        :return: encoded image (torch.tensor) of the shape [batch_size, *]
        """
        raise NotImplementedError


class BaseCaptionGenerator(torch.nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()

        self.vocabulary_size = vocabulary_size

    def freeze(self):
        """Sets the requires_grad parameter to False for some model parameters."""
        raise NotImplementedError

    def forward(self, encoded_image, caption_indices, *args):
        """Forward method.

        :param encoded_image: torch.tensor of the shape [batch_size, *] or None
        :param caption_indices: torch.tensor of the shape [batch_size, sequence_length] or None
        :param args: e.g., hidden state

        :return: output dict at least with 'logits' and 'indices' keys,
            where: logits is the torch.tensor of the shape [batch_size, vocabulary_size, sequence_length]
                   indices is the torch.tensor of the shape [batch_size, sequence_length]
        """
        raise NotImplementedError

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        """Generates caption indices like torch.tensor([1, 23, 5, 8, 2]).

        :param encoded_image: torch.tensor of the shape [1, *]
        :param sos_token_index: index of the "start of sequence" token (int)
        :param eos_token_index: index of the "end of sequence" token (int)
        :param max_length: maximum caption length (int)

        :return: caption indices (list of the length <= max_length)
        """
        raise NotImplementedError
