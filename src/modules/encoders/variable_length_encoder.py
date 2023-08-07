import torch
from torch import nn


class VariableLengthEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        requires_grad: bool = False
    ):
        super().__init__()
        self.rnn = nn.RNN(input_dim, embed_dim, bias=False, batch_first=True) \
                    .requires_grad_(requires_grad=requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rnn(x.transpose(-1, -2))[1].squeeze(0)
        return x
