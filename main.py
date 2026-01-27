import csv
import os
from datetime import datetime

import torch
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class ClassificatorImages:
    """Класс классификатора изображений."""

    def __init__(self):
        self.image_list = []             # список всех файлов jpg, jpeg, png найденных в /media
        self.classification_result = {}  # результат работы классификатора
        self.weights = MobileNet_V2_Weights.DEFAULT
        self.model = mobilenet_v2(weights=self.weights)
        self.model.eval()
        self.classify_classes = self.weights.meta["categories"]
        self.transform = self.weights.transforms()

    def collect_image(self):
        """
        Считывает все файлы из /media и
        присваивает значение атрибуту self.image_list.
        """
        files = os.listdir('./media')
        image_list = [file for file in files if file.split('.')[1] in ('jpg', 'jpeg', 'png')]
        self.image_list = image_list
        return image_list

    def classify_image(self, image_path, top_k=3):
        """
        Классифицирует одно изображение и возвращает класс
        из MobileNet_V2_Weights.DEFAULT.meta["categories"].
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img_t = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                output = self.model(img_t)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_probs, top_indices = torch.topk(probabilities,
                                                min(top_k, len(probabilities)))
            results = []
            for i in range(len(top_indices)):
                class_index = top_indices[i].item()
                results.append(self.classify_classes[class_index])
            return results
        except Exception as e:
            print(f'Ошибка при обработке {image_path}: {e}')
            return {'class_name': 'error'}

    def classify_all_images(self):
        """
        Классифицирует все изображения в папке /media,
        и добавляет результаты в self.classification_result.
        """
        for img in self.image_list:
            result = self.classify_image(f'./media/{img}')
            self.classification_result[img] = result
        return self.classification_result

    def save_to_csv(self, filename=None):
        """Сохраняет результаты классификации в CSV файл."""
        if not self.classification_result:
            print('Нет данных для сохранения. Сначала выполните классификацию.')
            return False
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"classification_results_{timestamp}.csv"
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Image Name',
                                 'Class1',
                                 'Class2',
                                 'Class3'])
                for image_name, classes in self.classification_result.items():
                    padded_classes = (classes + ['', '', ''])[:3]
                    writer.writerow([image_name] + padded_classes)
            print(f'Результаты сохранены в файл: {filename}')
            return True
        except Exception as e:
            print(f'Ошибка при сохранении в CSV: {e}')
            return False


if __name__ == '__main__':
    try:
        my_classificator = ClassificatorImages()  # создаем объект классификатора
        images = my_classificator.collect_image() # собираем все файлы из директории /media
        if images == []:
            raise ValueError('Нет изображений в папке /media.')
    except Exception as e:
        print(f'Ошибка: {e}')
    else:
        my_classificator.classify_all_images()          # классифицируем все собранные изображения
        my_classificator.save_to_csv()                  # охраняем результат в csv-файл
