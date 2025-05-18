import torch
import torch.nn as nn
import math
from einops import rearrange

from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator


class Model(BaseModel):
    def __init__(self, vocabulary, embedding_dim, hidden_dim, num_layers, num_heads, enc_emb_dim):
        super().__init__(vocabulary=vocabulary)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.enc_emb_dim = enc_emb_dim

        self.image_encoder = ImageEncoder(embedding_dim=self.embedding_dim)
        self.caption_generator = CaptionGenerator(
            vocabulary_size=len(self.vocabulary),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            enc_emb_dim = enc_emb_dim
        )


class ImageEncoder(BaseImageEncoder):
    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Load pretrained DINOv2 model
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False


    def freeze(self):
        pass

    def forward(self, image):
        x = nn.functional.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        features = self.backbone.get_intermediate_layers(x, n=1, reshape=True, return_class_token=True)[-1]
        return features[0]  # (batch, num_patches, embedding_dim)


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_layers, num_heads, enc_emb_dim=384, dropout=0.25):
        super().__init__(vocabulary_size=vocabulary_size)

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.to_decoder_emb = nn.Linear(enc_emb_dim, self.embedding_dim)
        self.to_logits = nn.Linear(embedding_dim, vocabulary_size)

    def freeze(self):
        pass

    def _get_embeddings(self, caption_indices):
        return self.embedding(caption_indices)

    def forward(self, encoded_image, caption_indices, hidden_state=None):
        # encoded_image: (batch, seq_len_img, embedding_dim)
        # caption_indices: (batch, seq_len_caption)

        image_embeddings = rearrange(encoded_image, "b c h w -> b (h w) c")
        image_embeddings = self.to_decoder_emb(image_embeddings)

        tgt_emb = self._get_embeddings(caption_indices)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        # Causal mask for decoder (tgt) to prevent attending to future tokens
        seq_len = caption_indices.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(caption_indices.device)

        # No mask for memory (image), attend fully
        memory = self.dropout(image_embeddings)

        output = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)

        logits = self.to_logits(output)
        logits = rearrange(logits, 'b s v -> b v s')

        return {'logits': logits, 'indices': logits.argmax(dim=-2)}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        caption_indices = [sos_token_index]

        for _ in range(max_length):
            input_tensor = torch.LongTensor(caption_indices).to(encoded_image.device).unsqueeze(0)
            output = self.forward(encoded_image=encoded_image, caption_indices=input_tensor)
            predicted_index = output['indices'][:, -1]
            caption_indices.append(predicted_index.item())
            if caption_indices[-1] == eos_token_index:
                break

        return caption_indices[1:]  # Drop SOS token


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)
