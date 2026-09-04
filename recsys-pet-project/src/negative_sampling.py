"""Popularity-based негативный сэмплинг.

Случайные негативы завышают метрики — редкий товар почти всегда "лёгкий"
негатив. Сэмплим пропорционально популярности в train, ближе к реальному
распределению показов.

NEG_RATIO фиксирован здесь одним местом — ALS и Two-Tower должны
использовать одно и то же соотношение 1:N, иначе сравнение в MLflow
(этап 6) будет некорректным.
"""

import numpy as np
import pandas as pd

NEG_RATIO = 4


class PopularityNegativeSampler:
    def __init__(self, train_positive: pd.DataFrame, item_map=None, seed: int = 0):
        item_counts = train_positive.groupby("item_id").size()
        if item_map is not None:
            item_counts = item_counts.reindex(item_map.raw_to_idx.keys()).fillna(0)
            self.items = np.array([item_map.raw_to_idx[i] for i in item_counts.index])
        else:
            self.items = item_counts.index.to_numpy()
        counts = item_counts.to_numpy(dtype=np.float64)
        self.probs = counts / counts.sum()
        self.rng = np.random.default_rng(seed)

    def sample(self, exclude: set, n: int) -> np.ndarray:
        """Сэмплит n негативов, пересэмплируя коллизии с `exclude` (позитивами)."""
        negatives = np.empty(n, dtype=self.items.dtype)
        filled = 0
        while filled < n:
            batch = self.rng.choice(self.items, size=(n - filled) * 2, p=self.probs, replace=True)
            batch = batch[[item not in exclude for item in batch]]
            take = min(len(batch), n - filled)
            negatives[filled:filled + take] = batch[:take]
            filled += take
        return negatives

    def sample_for_positive(self, exclude: set, ratio: int = NEG_RATIO) -> np.ndarray:
        return self.sample(exclude, ratio)
