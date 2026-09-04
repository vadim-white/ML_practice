"""Обёртка над implicit.als.AlternatingLeastSquares.

R ≈ U × Vᵀ: раскладываем sparse-матрицу взаимодействий на эмбеддинги
пользователей (U) и товаров (V). Confidence бинарный (1.0 на каждый
позитив), в духе implicit-feedback подхода этапа 1.
"""

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from src.id_mapping import IdMap


class ALSModel:
    def __init__(self, factors: int = 64, regularization: float = 0.01, iterations: int = 15, random_state: int = 0):
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
        )
        self.user_items = None

    def fit(self, train_positive_df, user_map: IdMap, item_map: IdMap) -> "ALSModel":
        user_idx = train_positive_df["user_id"].map(user_map.raw_to_idx).to_numpy()
        item_idx = train_positive_df["item_id"].map(item_map.raw_to_idx).to_numpy()
        confidence = np.ones(len(train_positive_df), dtype=np.float32)
        self.user_items = sp.csr_matrix(
            (confidence, (user_idx, item_idx)), shape=(len(user_map), len(item_map))
        )
        self.model.fit(self.user_items)
        return self

    @property
    def user_factors(self) -> np.ndarray:
        return self.model.user_factors

    @property
    def item_factors(self) -> np.ndarray:
        return self.model.item_factors

    def recommend_for_user(self, user_idx: int, k: int = 100, filter_already_liked: bool = True):
        item_idx, scores = self.model.recommend(
            user_idx,
            self.user_items[user_idx],
            N=k,
            filter_already_liked_items=filter_already_liked,
        )
        return item_idx, scores
