import torch
import os
from PIL import Image
from pprint import pprint
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


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

    def classify_image(self, image_path):
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
            best_prob, best_idx = torch.topk(probabilities, 1)
            class_index = best_idx.item()
            class_name = self.classify_classes[class_index]
            confidence = best_prob.item()
            return {'class_name': class_name,       # Конкретный класс из 1000
                    'class_index': class_index,     # Номер класса (0-999)
                    'confidence': confidence}       # вероятность (0-1)

        except Exception as e:
            print(f'Ошибка при обработке {image_path}: {e}')
            return {'class_name': 'error',
                    'class_index': -1,
                    'confidence': 0.0}

    def classify_all_images(self):
        """
        Классифицирует все изображения в папке /media,
        и добавляет результаты в self.classification_result.
        """
        for img in self.image_list:
            result = self.classify_image(f'./media/{img}')
            self.classification_result[img] = result
        return self.classification_result


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
        pprint(my_classificator.classification_result)  # распечатываем результат работы классификатора
