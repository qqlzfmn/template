import torch
from torch import nn

from src.modules import encoders, fusions


class Network(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoders = nn.ModuleDict({
            feat: encoder(dim, embed_dim, requires_grad=True) \
            for feat, (encoder, dim) in feature_dim_dict.items()
        })
        self.fuser = nn.Sequential(
            ConcatFusionModule(),
            nn.LayerNorm(embed_dim * len(feature_dim_dict)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * len(feature_dim_dict), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = {feat: encoder(x[feat].to(self.device)) \
            for feat, encoder in self.encoders.items()}
        x = self.fuser(x)
        x = self.classifier(x)
        return x

    def freeze(self, layer_name):
        for name, param in self.named_parameters():
            if name.startswith(layer_name):
                param.requires_grad = False
        return self
