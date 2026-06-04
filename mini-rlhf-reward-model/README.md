# Mini RLHF Reward Model

Исследование подходов к построению reward model в стиле RLHF на датасете [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf).

**Задача:** предсказать, какой из двух ответов ассистента предпочтительнее (chosen vs rejected).  
**Метрика:** pairwise accuracy на официальном test split (8 552 пары).

---

## Структура репозитория

```
mini-rlhf-reward-model/
├── nootebooks/
│   ├── baseline.ipynb                           # Эксп. 1 — LogReg + MiniLM
│   ├── reward_model_v2_with_embeddings.ipynb    # Эксп. 2 — PyTorch MLP (локально)
│   ├── reward_model_v2_gpu_colab_with_embeddings.ipynb  # Эксп. 3 — MLP + MPNet (GPU)
│   ├── final_model_pipeline_logreg.ipynb        # Эксп. 4 — LogReg + MPNet (160k)
│   ├── reward_model_finetune_colab_2.ipynb      # Эксп. 5 — RoBERTa fine-tune v1
│   ├── reward_model_finetune_colab_4_final.ipynb      # Эксп. 6 — RoBERTa fine-tune v2
│   └── reward_model_finetune_colab.ipynb        # Эксп. 7 — RoBERTa fine-tune v3 (актуальный)
│
├── data/embeddings/
│   ├── chosen_15k.npy / rejected_15k.npy        # MiniLM эмбеддинги, 15k пар
│   ├── chosen_full_mpnet.npy / rejected_full_mpnet.npy  # MPNet, 160k (в .gitignore)
│   └── chosen_test_mpnet.npy / rejected_test_mpnet.npy  # MPNet, test split
│
├── models/
│   ├── logreg_mpnet_augmented.pkl   # LogReg модель (Эксп. 4)
│   ├── reward_mlp_15k.pt            # PyTorch MLP (Эксп. 2)
│   ├── reward_mlp_mpnet_full.pt     # PyTorch MLP на MPNet (Эксп. 3)
│   ├── reward_model_bt_best-2.zip   # RoBERTa checkpoint (Эксп. 5, ~439MB, в .gitignore)
│   └── reward_model_bt_best-4.zip   # RoBERTa checkpoint (Эксп. 6, ~319MB, в .gitignore)
│
├── requirements.txt
└── README.md
```

---

## Эксперименты

### Эксперимент 1 — `baseline.ipynb`
**LogReg + MiniLM-L6-v2 (frozen embeddings)**

- Эмбеддинги: `all-MiniLM-L6-v2` (384d), заморожены
- Признаки: `diff = emb_chosen - emb_rejected`
- Данные: 15 000 пар → 30 000 с аугментацией (swap)
- Модель: Logistic Regression (sklearn)
- **Test Accuracy: 0.6273**

---

### Эксперимент 2 — `reward_model_v2_with_embeddings.ipynb`
**PyTorch MLP поверх MiniLM (локально, M1 CPU)**

- Эмбеддинги: `all-MiniLM-L6-v2` (384d), заморожены
- Признаки: `[chosen, rejected, diff, abs_diff, prod, cos_sim]` → 3841 dim
- Модель: MLP (3841 → 128 → 64 → 1) + Dropout + BCEWithLogitsLoss
- Данные: 15 000 пар, 30 эпох с early stopping
- **Val Accuracy: ~0.615–0.62** (оверфит, не превзошёл LogReg)

---

### Эксперимент 3 — `reward_model_v2_gpu_colab_with_embeddings.ipynb`
**PyTorch MLP + MPNet (Colab GPU)**

- Эмбеддинги: `all-mpnet-base-v2` (768d), заморожены
- Данные: 160 800 пар (полный train split)
- **Test Accuracy: 0.50** — провал (ошибка в пайплайне, путаница меток)

---

### Эксперимент 4 — `final_model_pipeline_logreg.ipynb`
**LogReg + MPNet (полный датасет)**

- Эмбеддинги: `all-mpnet-base-v2` (768d), заморожены
- Признаки: `diff = emb_chosen - emb_rejected`
- Данные: 160 800 пар → 321 600 с аугментацией
- Модель: Logistic Regression
- **Test Accuracy: 0.5673 | ROC-AUC: 0.601**
- Вывод: больше данных + мощнее эмбеддинги — не помогло. MPNet сглаживает тонкие различия.

---

### Эксперимент 5 — `reward_model_finetune_colab_2.ipynb`
**RoBERTa fine-tune end-to-end, Bradley-Terry loss (v1)**

- Модель: `roberta-base` (125M) + Linear head, все слои обучаемы
- Loss: Bradley-Terry — `-log(σ(r_chosen − r_rejected))`
- Данные: 40 000 пар
- GPU: T4 (Google Colab)
- Конфиг: 5 эпох, batch=16, grad_accum=4, fp16
- Сохранённая модель: `models/reward_model_bt_best-2.zip`
- **Test Accuracy: 0.6322**

---

### Эксперимент 6 — `reward_model_finetune_colab_4_final.ipynb`
**RoBERTa fine-tune + заморозка слоёв + early stopping (v2)**

- Модель: `roberta-base`, заморожены нижние 8/12 слоёв (trainable: 28M из 125M)
- Loss: Bradley-Terry
- Данные: 40 000 пар
- GPU: T4 (Google Colab)
- Конфиг: max 10 эпох, early stopping patience=2, batch=64, fp16
- Остановился на эпохе 6 (early stopping)
- Сохранённая модель: `models/reward_model_bt_best-4.zip`
- **Test Accuracy: 0.6303**

---

### Эксперимент 7 — `reward_model_finetune_colab.ipynb`
**RoBERTa fine-tune (актуальная версия, в процессе)**

- Все улучшения из экспериментов 5–6 + фикс `truncation_side='left'`
- `truncation_side='left'` — ключевой фикс: chosen/rejected отличаются в конце текста (финальный ответ ассистента), поэтому обрезать нужно начало диалога, а не хвост
- Конфиг: max 10 эпох, early stopping, batch=64, заморозка 8 слоёв, dropout=0.1

---

## Итоговые результаты

| # | Тетрадка | Подход | Данные | Test Acc |
|---|----------|--------|--------|----------|
| 1 | `baseline.ipynb` | LogReg + MiniLM (frozen) | 15k | 0.6273 |
| 2 | `reward_model_v2_with_embeddings.ipynb` | MLP + MiniLM (frozen) | 15k | ~0.615 val |
| 3 | `reward_model_v2_gpu_colab_with_embeddings.ipynb` | MLP + MPNet (frozen) | 160k | 0.50 (fail) |
| 4 | `final_model_pipeline_logreg.ipynb` | LogReg + MPNet (frozen) | 160k | 0.5673 |
| 5 | `reward_model_finetune_colab_2.ipynb` | **RoBERTa BT-loss** | 40k | **0.6322** |
| 6 | `reward_model_finetune_colab_4_final.ipynb` | RoBERTa + layer freeze | 40k | 0.6303 |

---

## Выводы

**1. Fine-tuning encoder даёт лучший результат**  
RoBERTa fine-tune (0.6322) > LogReg baseline (0.6273). Но разрыв небольшой — задача объективно сложная.

**2. Frozen embeddings — потолок**  
MiniLM и MPNet не улавливают тонкие различия между chosen/rejected. Линейная модель поверх них быстро упирается в предел качества эмбеддингов.

**3. Больше данных ≠ лучше при слабом сигнале**  
LogReg на 160k (MPNet) дал 0.567 — хуже чем на 15k (MiniLM). Шум масштабируется вместе с данными.

**4. Главный инсайт про truncation**  
В датасете hh-rlhf тексты — это полные диалоги, где chosen и rejected отличаются только **в последнем ответе ассистента** (в хвосте строки). При `truncation='right'` (дефолт) токенизатор обрезал именно этот хвост — и модель обучалась на идентичных началах диалогов. Фикс: `tokenizer.truncation_side = 'left'`.

**5. Задача у потолка без LLM**  
0.63–0.64 — реалистичный ceiling для энкодерных моделей на hh-rlhf. Настоящий прорыв требует fine-tuning полноценного LLM (GPT-2+) или contrastive learning на парах.

---

## Технологии

- Python 3.11 / 3.12
- HuggingFace `datasets`, `transformers`
- `sentence-transformers`
- PyTorch (fp16, gradient checkpointing)
- scikit-learn
- Google Colab T4 GPU

---

## Автор

[Vadim K]
