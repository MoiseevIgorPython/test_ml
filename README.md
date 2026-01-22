# Классификатор изображений
Автор: Моисеев Игорь

Классификатор изображений с использованием MobileNetV2 и PyTorch.

## Возможности
- Автоматический сбор изображений из папки `media`
- Классификация с помощью предобученной модели MobileNetV2
- Вывод результатов в формате:

```
{'filename.jpg': ['top1_classname', 'top2_classname', 'top3_classname'],}
```
## Структура проекта

```text
test_ml/
├── main.py             # Основной файл с классом классификатора
├── media/              # Папка для изображений
├── requirements.txt    # Зависимости
└── README.md           # Документация
```
## Установка

1. Клонируйте репозиторий:
```
git clone https://github.com/MoiseevIgorPython/test_ml.git
```

Тестовые изображения загружены в репозиторий в директорию media

Установите зависимости:

```
pip install -r requirements.txt
```
## Использование:

```
python main.py
```
Результат работы скрипта сохраняется в файл вида:

```
classification_result_date_time.csv
```
