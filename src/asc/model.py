import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel

class AspectClassifier(nn.Module):
    def __init__(self, embed, dropout=0.5, num_classes=4, pool="max"):
        super().__init__()
        assert pool in ("max", "mean")
        self.embed = embed
        self.embed_dropout = nn.Dropout(dropout)
        if hasattr(embed, "embedding_dim"):
            embed_size = embed.embedding_dim
        else:
            embed_size = embed.config.hidden_size
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_size, num_classes),
        )
        self.pool = pool

    def forward(self, tokens, aspect_mask):
        """

        :param tokens:
        :param aspect_mask: bsz x max_len, 1 for aspect
        :return:
        """
        tokens = self.embed(tokens, token_type_ids=None)  # bsz x max_len x hidden_size

        # if isinstance(tokens, tuple):
        #     tokens = tokens[0]
        tokens = self.embed_dropout(tokens)

        aspect_mask = aspect_mask.eq(1)
        if self.pool == "mean":
            tokens = tokens.masked_fill(aspect_mask.unsqueeze(-1).eq(0), 0)
            tokens = tokens.sum(dim=1)
            preds = tokens / aspect_mask.sum(dim=1, keepdims=True).float()
        elif self.pool == "max":
            aspect_mask = aspect_mask.unsqueeze(-1).eq(0)  # bsz x max_len x 1
            tokens = tokens.masked_fill(aspect_mask, -10000.0)
            preds, _ = tokens.max(dim=1)
        preds = self.ffn(preds)
        return preds
