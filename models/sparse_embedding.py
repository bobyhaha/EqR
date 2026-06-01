import torch
from torch import nn

from models.common import trunc_normal_init_


class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype) -> None:
        super().__init__()
        self.cast_to = cast_to
        self.weights = nn.Buffer(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std),
            persistent=True,
        )
        self.local_weights = nn.Buffer(torch.zeros(batch_size, embedding_dim, requires_grad=True), persistent=False)
        self.local_ids = nn.Buffer(torch.zeros(batch_size, dtype=torch.int32), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        real_bs = inputs.shape[0]
        indices = inputs.to(torch.long)
        if not self.training or real_bs != self.local_weights.shape[0]:
            return self.weights[indices].to(self.cast_to)

        with torch.no_grad():
            self.local_weights.copy_(self.weights[indices])
            self.local_ids.copy_(inputs)
        return self.local_weights.to(self.cast_to)
