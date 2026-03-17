# Этап 4-2. Обучение модели YOLOv8 на датасете Minecraft
import os
from ultralytics import YOLO

def train_yolo():
    # Путь к конфигурации данных
    data_yaml = 'datasets/minecraft/data.yaml'
    # Базовая модель для дообучения
    model_path = 'yolov8s.pt'
    
    if not os.path.exists(data_yaml):
        print(f"Ошибка: Файл {data_yaml} не найден.")
        return

    # Инициализация модели
    print(f"Загрузка модели {model_path}...")
    model = YOLO(model_path)

    # Запуск обучения
    # project='artifacts' и name='yolo' направят все результаты в artifacts/yolo
    # imgsz=512 - согласно рекомендациям по эффективности
    # epochs=25 - YOLO учится быстро, можно взять чуть больше для лучшего качества
    print("Запуск обучения YOLOv8...")
    results = model.train(
        data=data_yaml,
        epochs=25,
        imgsz=512,
        batch=16, # Можно увеличить batch, если позволяет память GPU
        project='artifacts',
        name='yolo',
        exist_ok=True, # Перезаписывать в ту же папку, если она существует
        optimizer='SGD', # Стандартный для детекции
        lr0=0.01,
        pretrained=True
    )

    print(f"Обучение YOLO завершено. Результаты сохранены в artifacts/yolo")

if __name__ == '__main__':
    train_yolo()
