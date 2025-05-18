import torch.nn
import math

from einops import rearrange, pack

from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator


class Model(BaseModel):
    def __init__(self, vocabulary, embedding_dim, hidden_dim, num_layers, num_heads):
        super().__init__(vocabulary=vocabulary)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.image_encoder = ImageEncoder(embedding_dim=self.embedding_dim)
        self.caption_generator = CaptionGenerator(vocabulary_size=len(self.vocabulary),
                                                  embedding_dim=self.embedding_dim,
                                                  hidden_dim=self.hidden_dim,
                                                  num_heads=self.num_heads,
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
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_layers, num_heads, dropout=0.25):
        super().__init__(vocabulary_size=vocabulary_size)
        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(embedding_dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_logits = torch.nn.Linear(embedding_dim, vocabulary_size)

    def freeze(self):
        pass

    def _get_embeddings(self, caption_indices):
        return self.embedding(caption_indices)

    def forward(self, encoded_image, caption_indices, hidden_state=None):
        # encoded_image: (batch, embedding_dim)
        # caption_indices: (batch, seq_len)

        # Remove <SOS> token
        if caption_indices is not None:
            caption_indices = caption_indices[:, 1:]

        word_embeds = self.embedding(caption_indices)  # (batch, seq_len, embed_dim)
        batch_size = word_embeds.size(0)

        # Add image token at the start
        encoded_image = encoded_image.unsqueeze(1)  # (batch, 1, embed_dim)
        tokens = torch.cat([encoded_image, word_embeds], dim=1)  # (batch, 1+seq_len, embed_dim)

        tokens = self.pos_encoder(tokens)
        tokens = self.dropout(tokens)

        tokens = tokens.transpose(0, 1)  # Transformer expects (seq_len, batch, embed_dim)

        # Apply causal mask
        seq_len = tokens.size(0)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(tokens.device)

        output = self.transformer(tokens, mask=causal_mask)  # (seq_len, batch, embed_dim)
        output = output.transpose(0, 1)  # (batch, seq_len, embed_dim)

        logits = self.to_logits(output)  # (batch, seq_len, vocab_size)

        logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')

        return {'logits': logits, 'indices': logits.argmax(dim=-2)}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        caption_indices = [sos_token_index]

        for _ in range(max_length):
            output = self.forward(encoded_image=encoded_image,
                                  caption_indices=torch.LongTensor(caption_indices).to(encoded_image.device).unsqueeze(dim=0))
            
            predicted_index = output['indices'][:, -1]

            caption_indices.append(predicted_index.item())
            if caption_indices[-1] == eos_token_index:
                break

        return caption_indices[1:]  # drop SOS token

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x