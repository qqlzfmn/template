import torch
from torch import nn


class FixedLengthEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        requires_grad: bool = False
    ):
        super().__init__()
        self.fc = nn.Linear(input_dim, embed_dim, bias=False) \
                    .requires_grad_(requires_grad=requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x
