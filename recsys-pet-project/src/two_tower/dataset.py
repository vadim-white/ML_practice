"""Torch Dataset для Two-Tower: (user_idx, pos_item_idx, neg_item_idx[NEG_RATIO]),
негативы — тот же popularity-based сэмплер, что и у остального этапа 2."""

import torch
from torch.utils.data import Dataset

from src.negative_sampling import NEG_RATIO, PopularityNegativeSampler


class TwoTowerDataset(Dataset):
    def __init__(self, train_positive_df, user_map, item_map, neg_ratio: int = NEG_RATIO, seed: int = 0):
        self.user_idx = train_positive_df["user_id"].map(user_map.raw_to_idx).to_numpy()
        self.item_idx = train_positive_df["item_id"].map(item_map.raw_to_idx).to_numpy()
        self.sampler = PopularityNegativeSampler(train_positive_df, item_map=item_map, seed=seed)
        self.neg_ratio = neg_ratio

    def __len__(self) -> int:
        return len(self.user_idx)

    def __getitem__(self, i: int):
        pos_item = self.item_idx[i]
        negatives = self.sampler.sample_for_positive(exclude={pos_item}, ratio=self.neg_ratio)
        return (
            torch.tensor(self.user_idx[i], dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long),
        )
