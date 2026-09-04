"""Raw ID MovieLens не сплошные — строим свой dense-индекс для ALS-матрицы и embedding-таблиц Two-Tower."""

import pandas as pd


class IdMap:
    def __init__(self, raw_ids: pd.Series):
        unique_ids = pd.unique(raw_ids)
        self.raw_to_idx = {raw_id: idx for idx, raw_id in enumerate(unique_ids)}
        self.idx_to_raw = {idx: raw_id for raw_id, idx in self.raw_to_idx.items()}

    def __len__(self):
        return len(self.raw_to_idx)

    def to_idx(self, raw_ids):
        return [self.raw_to_idx[raw_id] for raw_id in raw_ids]

    def to_raw(self, indices):
        return [self.idx_to_raw[idx] for idx in indices]

    def contains(self, raw_id) -> bool:
        return raw_id in self.raw_to_idx


def build_id_maps(train_df: pd.DataFrame) -> tuple[IdMap, IdMap]:
    user_map = IdMap(train_df["user_id"])
    item_map = IdMap(train_df["item_id"])
    return user_map, item_map
