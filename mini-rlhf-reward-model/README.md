# Mini RLHF Reward Model

## 📖 Описание проекта

Этот проект представляет собой исследование по созданию reward model в стиле RLHF (Reinforcement Learning from Human Feedback). 

**Цель:** научиться предсказывать, какой из двух ответов предпочтительнее человеку, используя эмбеддинги текстов и машинное обучение.

В ходе проекта были проэкерементированы различные подходы — от классических ML-моделей до нейронных сетей — и сделаны важные выводы о применимости разных методов для задачи ранжирования текстовых предпочтений.

---

## 🔬 Проведённые эксперименты

### Эксперимент 1: Baseline (Logistic Regression + MiniLM)
- **Модель:** Logistic Regression (scikit-learn)
- **Эмбеддинги:** `all-MiniLM-L6-v2` (384 измерения)
- **Признаки:** `diff = emb_chosen - emb_rejected`
- **Данные:** 1,000 пар из train-сплита
- **Результат:** **Accuracy ≈ 0.64** (на валидации)

✅ **Вывод:** Простая линейная модель показала хороший результат для старта.

---

### Эксперимент 2: PyTorch MLP (15k данных)
- **Модель:** MLP (1921 → 128 → 64 → 1) с Dropout
- **Эмбеддинги:** `all-MiniLM-L6-v2`
- **Признаки:** `[chosen, rejected, diff, abs_diff, prod, cos_sim]` (1921 признак)
- **Данные:** 15,000 пар + аугментация (swap)
- **Результат:** **Val Accuracy ≈ 0.57-0.62**, сильное переобучение

⚠️ **Вывод:** Нейросеть переобучается на малых данных, не обобщает лучше LogReg.

---

### Эксперимент 3: Масштабирование + all-mpnet-base-v2
- **Модель:** Logistic Regression + MLP (сравнение)
- **Эмбеддинги:** `all-mpnet-base-v2` (768 измерений, более качественные)
- **Данные:** **весь train-сплит (160,800 пар)**
- **Результаты:**
  - **LogReg + mpnet:** Test Accuracy ≈ **0.567**, ROC-AUC ≈ **0.601**
  - **MLP + mpnet:** Test Accuracy ≈ **0.50-0.57**, нестабильно

❌ **Вывод:** 
1. Больше данных ≠ лучше результат при слабом сигнале
2. Более мощные эмбеддинги (`mpnet`) **сглаживают** тонкие различия между ответами
3. `all-MiniLM-L6-v2` (меньшая модель) лучше ловит локальные различия в `hh-rlhf`

---

### Эксперимент 4: Siamese Reward Scorer (PyTorch)
- **Архитектура:** Две отдельные сети для scoring chosen/rejected
- **Loss:** MarginRankingLoss / BCEWithLogitsLoss
- **Результат:** Не превзошёл простой diff + LogReg

❌ **Вывод:** Сложные архитектуры не дают преимущества при отсутствии чёткого сигнала в данных.

---

## 🎯 Итоговые выводы

| Подход | Accuracy | Вывод |
|--------|----------|-------|
| **LogReg + MiniLM (1k)** | **0.640** | ✅ **Лучший результат**, стабильный, интерпретируемый |
| LogReg + mpnet (160k) | 0.567 | Хуже, несмотря на больше данных и мощнее эмбеддинги |
| MLP + MiniLM (15k) | 0.57-0.62 | Переобучение, не обобщает |
| MLP + mpnet (160k) | 0.50-0.57 | Нестабильно, близко к рандому |

### 🔑 Ключевые инсайты:

1. **Простота побеждает сложность**  
   Logistic Regression с простыми признаками (`diff`) превзошла все нейросетевые подходы.

2. **Больше данных ≠ лучше**  
   На задаче с тонкими различиями между ответами, масштабирование до 160k не помогло — шум накапливается.

3. **Выбор эмбеддингов критичен**  
   `all-MiniLM-L6-v2` (384 dim) оказался **лучше** `all-mpnet-base-v2` (768 dim) для этой конкретной задачи, потому что меньшая модель меньше "сглаживает" тонкие различия.

4. **Задача сложная**  
   Даже лучший результат (0.64) — это лишь немного выше случайного угадывания. Это показывает, что preferences в `hh-rlhf` — очень тонкая задача, требующая либо fine-tuned LLM-as-a-judge, либо contrastive fine-tuning самих эмбеддингов.

---

## 📁 Структура проекта

```
mini-rlhf-reward-model/
├── notebooks/
│   ├── baseline.ipynb              # 🔵 LogReg + MiniLM (1k) → 0.64
│   ├── final_model_pipeline.ipynb  # 🟢 LogReg + mpnet (160k) → 0.567
│   ├── reward_model_v2.ipynb       # 🟡 PyTorch MLP эксперименты
│   └── reward_model_v2_gpu_c...    # 🟡 PyTorch MLP на GPU
│
├── data/embeddings/
│   ├── chosen_15k.npy              # Эмбеддинги (MiniLM, 15k)
│   ├── rejected_15k.npy
│   ├── chosen_full_mpnet.npy       # Эмбеддинги (mpnet, 160k)
│   ├── rejected_full_mpnet.npy
│   ├── chosen_test_mpnet.npy       # Test set эмбеддинги
│   └── rejected_test_mpnet.npy
│
├── models/
│   ├── logreg_mpnet_augmente...    # Финальная LogReg модель
│   ├── reward_mlp_15k.pt           # PyTorch MLP (15k)
│   └── reward_mlp_mpnet_full.pt    # PyTorch MLP (160k)
│
├── README.md
└── requirements.txt
```

### 📓 Описание ноутбуков:

#### **baseline.ipynb** (Logistic Regression)
- Начальный эксперимент с 1,000 пар
- `all-MiniLM-L6-v2` + diff-признаки
- **Лучший результат проекта: 0.64 accuracy**
- Простой, стабильный, интерпретируемый пайплайн

#### **final_model_pipeline.ipynb** (Logistic Regression)
- Полный пайплайн на всём датасете (160k)
- `all-mpnet-base-v2` + аугментация
- Test accuracy: **0.567**
- Показывает, что масштабирование не всегда помогает

#### **reward_model_v2.ipynb** и **reward_model_v2_gpu_c...** (PyTorch)
- Эксперименты с нейронными сетями:
  - MLP с разными архитектурами
  - Siamese Reward Scorer
  - Разные loss-функции (BCE, MarginRanking)
  - BatchNorm, Dropout, LayerNorm
- Результаты: переобучение, нестабильность, accuracy 0.50-0.62
- Важный негативный результат: сложные модели не всегда лучше

---

##  Финальный пайплайн

**Почему выбрана Logistic Regression:**

1. ✅ **Лучшая метрика:** 0.56 accuracy (выше всех нейросетей)
2. ✅ **Стабильность:** Не переобучается, хорошо обобщает
3. ✅ **Интерпретируемость:** Можно посмотреть `clf.coef_` и понять, какие измерения важны
4. ✅ **Скорость:** Обучение за секунды, инференс мгновенный
5. ✅ **Простота:** Минимум кода, легко поддерживать

**Пайплайн:**
```
Текст → all-mpnet-base-v2 → emb_chosen, emb_rejected

         ↓
diff = emb_chosen - emb_rejected (768 признаков)

         ↓
LogisticRegression.predict(diff) → 1 (chosen better) / 0 (rejected better)
```

---

##  Быстрый старт

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Запуск baseline
1. Откройте `notebooks/baseline.ipynb`
2. Запустите все ячейки
3. Получите accuracy ~0.64 на валидации



## 📊 Результаты

| Модель | Эмбеддинги | Данные | Test Accuracy | ROC-AUC |
|--------|-----------|--------|--------------|---------|
| **LogReg (baseline)** | all-MiniLM-L6-v2 | 1k pairs | **0.640** | - |
| LogReg (full) | all-mpnet-base-v2 | 160k pairs | 0.567 | 0.601 |
| MLP (PyTorch) | all-MiniLM-L6-v2 | 15k pairs | 0.57-0.62 | - |
| MLP (PyTorch) | all-mpnet-base-v2 | 160k pairs | 0.50-0.57 | - |

---


## 📚 Используемые технологии

- **Python 3.11**
- **HuggingFace Datasets** — загрузка `Anthropic/hh-rlhf`
- **Sentence-Transformers** — эмбеддинги текстов
- **Scikit-learn** — Logistic Regression
- **PyTorch** — эксперименты с MLP
- **NumPy/Pandas** — работа с данными
- **Matplotlib** — визуализация

---


## 👤 Автор

[Vadim K]

## 📄 Лицензия

MIT License
