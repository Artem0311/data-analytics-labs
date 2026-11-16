# Лабораторна робота №4 — Model Evaluation and Refinement

## Мета роботи

Отримати навички оцінки якості моделі.

## Виконані завдання

1. Продемонстровано перенавчання (Overfitting) за допомогою поліноміальної регресії дев'ятого ступеня та побудовано графік.
2. Продемонстровано недонавчання (Underfitting) за допомогою моделі з постійним прогнозом та побудовано графік.
3. За допомогою GridSearchCV підібрано оптимальний гіперпараметр alpha для Ridge Regression.

## Результати

| Завдання | Результат |
|---|---|
| Overfitting | R² train = 1.0000; R² test = -1507965243937.7581 |
| Underfitting | R² train = 0.0000; R² test = -0.0543 |
| Ridge Regression | Найкраще alpha = 10 |
| Cross-Validation | R² = 0.7651 |
| Тестування Ridge | R² = 0.7669 |

## Використані технології

- Python
- pandas
- NumPy
- matplotlib
- scikit-learn

## Набір даних

UCI Automobile Dataset

https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data

## Запуск програми

З кореневої папки репозиторію:

```bash
source .venv/bin/activate
python lab-04/main.py
```

## Файли лабораторної роботи

- `main.py` — програма для виконання трьох завдань.
- `overfitting.png` — графік перенавчання.
- `underfitting.png` — графік недонавчання.
- `Дата аналітика, Лабораторна робота 4.pdf` — звіт.
