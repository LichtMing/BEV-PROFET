import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class TimeEmbeddingSine(nn.Module):
    def __init__(self,
                 d_model: int = 64,
                 temperature: int = 10000,
                 scale: Optional[float] = None,
                 requires_grad: bool = False):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.scale = 2 * math.pi if scale is None else scale
        self.requires_grad = requires_grad

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs.clone()
        d_embed = self.d_model
        dim_t = torch.arange(d_embed, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / d_embed)
        x = x / dim_t
        x = torch.stack((x[..., 0::2].sin(), x[..., 1::2].cos()), dim=-1).flatten(-2)
        return x if self.requires_grad else x.detach()


class TimeEmbeddingLearnable(nn.Module):
    def __init__(self,
                 max_size: int = 7,
                 d_model: int = 64):
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embedding(max_size, d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.embeddings(inputs).flatten(-2)