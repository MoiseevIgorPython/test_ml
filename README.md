# Классификатор изображений
Автор: Моисеев Игорь

Простой классификатор изображений с использованием MobileNetV2 и PyTorch.

## Возможности
- Автоматический сбор изображений из папки `media`
- Классификация с помощью предобученной модели MobileNetV2
- Красивый вывод результатов в формате:

```
{'filename.jpg': {'class_index': int,
                 'class_name': 'Class Name',
                 'confidence': float},
}
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
Использование

```
python main.py
```

Структура проекта

```text
test_ml/
├── main.py             # Основной файл с классом классификатора
├── media/              # Папка для изображений
├── requirements.txt    # Зависимости
└── README.md           # Документация
```