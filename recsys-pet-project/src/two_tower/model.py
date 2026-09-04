"""Two-Tower retrieval модель.

Две башни кодируют user/item id в эмбеддинги одной размерности. Обучение —
in-batch InfoNCE (тот же принцип, что для текстовых эмбеддеров в RAG-блоке,
только применённый к id): сближает эмбеддинг пользователя с товарами, с
которыми было взаимодействие, и отдаляет от остальных товаров в батче плюс
явных popularity-based негативов.
"""

import torch
import torch.nn.functional as F
from torch import nn


class Tower(nn.Module):
    def __init__(self, num_ids: int, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_ids, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids)
        return F.normalize(x + self.mlp(x), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, temperature: float = 0.1):
        super().__init__()
        self.user_tower = Tower(num_users, embedding_dim)
        self.item_tower = Tower(num_items, embedding_dim)
        self.temperature = temperature

    def forward(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor, neg_item_ids: torch.Tensor) -> torch.Tensor:
        """InfoNCE loss: позитивы на диагонали, негативы — товары в батче
        плюс собственные popularity-сэмплированные негативы каждого примера."""
        user_emb = self.user_tower(user_ids)
        pos_emb = self.item_tower(pos_item_ids)
        neg_emb = self.item_tower(neg_item_ids.reshape(-1)).reshape(neg_item_ids.shape[0], neg_item_ids.shape[1], -1)

        pos_scores = (user_emb * pos_emb).sum(-1, keepdim=True) / self.temperature
        neg_scores = torch.einsum("bd,bnd->bn", user_emb, neg_emb) / self.temperature
        in_batch_scores = user_emb @ pos_emb.T / self.temperature

        logits = torch.cat([pos_scores, neg_scores, in_batch_scores], dim=1)
        labels = torch.zeros(user_ids.shape[0], dtype=torch.long, device=user_ids.device)
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def user_embeddings(self, user_ids: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_ids)

    @torch.no_grad()
    def item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_ids)
