# Mini RLHF Reward Model (Baseline)

## Описание проекта

Этот проект представляет собой упрощённую реализацию reward model в стиле RLHF (Reinforcement Learning from Human Feedback).

Цель — научиться предсказывать, какой из двух ответов предпочтительнее человеку, используя эмбеддинги текстов и классический ML.


## Что реализовано на данном этапе

### 1. Работа с датасетом
Использован датасет:
- Anthropic/hh-rlhf

Каждый пример содержит:
- chosen — предпочтительный ответ
- rejected — менее предпочтительный ответ


### 2. Преобразование текста в эмбеддинги
Использована модель:
- SentenceTransformer
- all-MiniLM-L6-v2

Текст преобразуется в вектор размерности 384.



### 3. Построение признаков
Для каждой пары ответов вычисляется:

diff = embedding(chosen) - embedding(rejected)



### 4. Обучение модели
Использована модель:
- Logistic Regression (scikit-learn)



### 5. Оценка качества
Метрика:
- Accuracy

Результат:
- Accuracy ≈ 0.64




## Используемые технологии

- Python 3.11
- HuggingFace Datasets
- Sentence-Transformers
- Scikit-learn
- NumPy
- Pandas



## Статус проекта

Baseline completed
Ready for improvements
