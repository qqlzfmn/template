import torch
from torch import nn
from typing import Dict


class ConcatFusionModule(nn.Module):
    """Module to fuse modalities via concatenation. Sorted by keys for consistency.

    Inputs:
        embeddings (Dict[str, Tensor]): A dictionary mapping modalities to their
            tensor representations.

    """

    def __init__(self, projection: nn.Module = None):
        super().__init__()
        if projection:
            self.projection = projection
        else:
            self.projection = nn.Identity()

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        concatenated_in = torch.cat(
            [embeddings[k] for k in sorted(embeddings.keys())], dim=-1
        )
        return self.projection(concatenated_in)
