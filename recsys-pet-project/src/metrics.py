"""Метрики retrieval/ranking, общие для этапов 2-4.

recall_at_k/ndcg_at_k/evaluate_fixed_recommendation перенесены как есть из
ноутбука 01 (popularity baseline). evaluate_recommendations — персонализированный
вариант для этапа 2 и далее, где у каждого пользователя свой top-k список,
а не один общий на всех.
"""

import numpy as np
import pandas as pd


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return np.nan
    top_k = set(recommended[:k])
    return len(top_k & relevant) / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return np.nan
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else np.nan


def evaluate_fixed_recommendation(recommended: list, test_positive: pd.DataFrame, k: int) -> dict:
    """Оценка НЕ-персонализированного списка (одного и того же для всех пользователей)."""
    relevant_by_user = test_positive.groupby("user_id")["item_id"].apply(set)
    recalls = relevant_by_user.apply(lambda relevant: recall_at_k(recommended, relevant, k))
    ndcgs = relevant_by_user.apply(lambda relevant: ndcg_at_k(recommended, relevant, k))
    return {"recall@k": recalls.mean(), "ndcg@k": ndcgs.mean(), "n_users": len(relevant_by_user)}


def evaluate_recommendations(recommended_by_user: dict, test_positive: pd.DataFrame, k: int) -> dict:
    """Оценка персонализированных списков (свой top-k на каждого пользователя),
    используется для ALS/Two-Tower retrieval начиная с этапа 2."""
    relevant_by_user = test_positive.groupby("user_id")["item_id"].apply(set)
    recalls = relevant_by_user.index.map(
        lambda user_id: recall_at_k(recommended_by_user.get(user_id, []), relevant_by_user[user_id], k)
    )
    ndcgs = relevant_by_user.index.map(
        lambda user_id: ndcg_at_k(recommended_by_user.get(user_id, []), relevant_by_user[user_id], k)
    )
    return {
        "recall@k": pd.Series(recalls).mean(),
        "ndcg@k": pd.Series(ndcgs).mean(),
        "n_users": len(relevant_by_user),
    }
