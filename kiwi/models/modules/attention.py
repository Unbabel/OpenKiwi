import torch
from torch import nn


class Attention(nn.Module):
    """Generic Attention Implementation.
       Module computes a convex combination of a set of values based on the fit
       of their keys with a query.
    """

    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer
        self.mask = None

    def forward(self, query, keys, values=None):
        if values is None:
            values = keys
        scores = self.scorer(query, keys)
        # Masked Softmax
        scores = scores - scores.mean(1, keepdim=True)  # numerical stability
        scores = torch.exp(scores)
        if self.mask is not None:
            scores = self.mask * scores
        convex = scores / scores.sum(1, keepdim=True)
        return torch.einsum('bs,bsi->bi', [convex, values])

    def set_mask(self, mask):
        self.mask = mask
