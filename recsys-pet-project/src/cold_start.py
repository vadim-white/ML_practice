"""Cold-start fallback для retrieval.

Пользователей без train-истории (~35.9% test, этап 1) подменяем popularity
ranking'ом вместо модельного скора. Cold-start товары решаются раньше по
пайплайну — их просто не upsert'им в vector store.
"""

import pandas as pd


def retrieve_candidates(user_id, model_score_fn, popularity_ranking: pd.DataFrame, known_users: set, k: int = 100) -> list:
    if user_id not in known_users:
        return popularity_ranking.head(k).index.tolist()
    return model_score_fn(user_id, k)
