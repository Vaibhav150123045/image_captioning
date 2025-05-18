import torch.nn
# import torchvision.models
# from transformers import AutoImageProcessor, AutoModel

from einops import rearrange, pack

from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator


class Model(BaseModel):
    def __init__(self, vocabulary, embedding_dim, hidden_dim, num_layers):
        super().__init__(vocabulary=vocabulary)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.image_encoder = ImageEncoder(embedding_dim=self.embedding_dim)
        self.caption_generator = CaptionGenerator(vocabulary_size=len(self.vocabulary),
                                                  embedding_dim=self.embedding_dim,
                                                  hidden_dim=self.hidden_dim,
                                                  num_layers=self.num_layers)


class ImageEncoder(BaseImageEncoder):
    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Load pretrained DINOv2 model
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14')

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Project to target embedding dim
        self.projector = torch.nn.Sequential(
            torch.nn.LayerNorm(self.backbone.embed_dim),
            torch.nn.Linear(self.backbone.embed_dim, self.embedding_dim)
        )

    def freeze(self):
        # Already frozen in init
        pass

    def forward(self, image):
        x = torch.nn.functional.interpolate(image, size=(
            224, 224), mode="bilinear", align_corners=False)
        out = self.backbone.get_intermediate_layers(x, n=1, reshape=True, return_class_token=True)[0]
        return self.projector(out[1])


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_layers):
        super().__init__(vocabulary_size=vocabulary_size)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.embedding_dim),
            torch.nn.Dropout(0.5)
        )

        self.rnn = torch.nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True
        )

        self.init_h = torch.nn.Linear(embedding_dim, hidden_dim * num_layers)
        self.init_c = torch.nn.Linear(embedding_dim, hidden_dim * num_layers)

        self.to_logits = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.vocabulary_size)

    def freeze(self):
        pass

    def _get_embeddings(self, caption_indices):
        return self.embedding(caption_indices)

    def _init_hidden_state(self, encoded_image):
        # encoded_image: [batch, embedding_dim]
        h0 = self.init_h(encoded_image)
        c0 = self.init_c(encoded_image)

        # Reshape to [num_layers, batch, hidden_dim]
        batch_size = encoded_image.size(0)
        h0 = h0.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2)
        c0 = c0.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2)

        return (h0.contiguous(), c0.contiguous())

    def forward(self, encoded_image, caption_indices, hidden_state=None):
        if encoded_image is not None and hidden_state is None:
            hidden_state = self._init_hidden_state(encoded_image)

        embeddings = self._get_embeddings(caption_indices)

        output, hidden_state = self.rnn(input=embeddings, hx=hidden_state)
        logits = self.to_logits(output)

        logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')

        return {'logits': logits, 'indices': logits.argmax(dim=-2), 'hidden_state': hidden_state}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        caption_indices = []

        hidden_state = self._init_hidden_state(encoded_image)

        current_input = torch.tensor([[sos_token_index]], device=encoded_image.device)

        for _ in range(max_length):
            embedding = self._get_embeddings(current_input)  # shape: [1, 1, embedding_dim]
            output, hidden_state = self.rnn(embedding, hidden_state)

            logits = self.to_logits(output.squeeze(1))  # shape: [1, vocab_size]
            predicted_index = logits.argmax(dim=-1)  # shape: [1]

            caption_indices.append(predicted_index.item())

            if predicted_index.item() == eos_token_index:
                break

            current_input = predicted_index.unsqueeze(0)  # [1, 1]

        return caption_indices

