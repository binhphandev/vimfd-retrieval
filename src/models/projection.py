import torch
import torch.nn as nn
import torch.nn.functional as F
from config import EMBED_DIM, PROJECTION_DROPOUT    


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
            nn.GELU(),
            nn.Dropout(PROJECTION_DROPOUT),
            nn.Linear(EMBED_DIM, EMBED_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1)  # L2 normalize the output embeddings   