import os
import glob
from ultralytics import YOLO
import cv2
from PIL import Image

def run_yolo_inference():
    # 1. Настройка путей
    model_path = 'artifacts/yolo/weights/best.pt'
    test_images_dir = 'datasets/minecraft/images/test'
    output_dir = 'artifacts/inference/yolo'
    
    if not os.path.exists(model_path):
        print(f"Ошибка: Веса {model_path} не найдены. Сначала запустите обучение YOLO.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 2. Инициализация модели
    print(f"Загрузка обученной модели YOLO: {model_path}...")
    model = YOLO(model_path)

    # 3. Прогон по тестовым изображениям
    image_paths = glob.glob(os.path.join(test_images_dir, '*.jpg'))
    print(f"Найдено {len(image_paths)} изображений для инференса.")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        
        # Инференс с порогом уверенности 0.25
        results = model.predict(img_path, conf=0.25, save=False)
        
        for result in results:
            # Отрисовка результатов (plot возвращает BGR)
            res_plotted = result.plot()
            # Конвертация в RGB для корректного сохранения через PIL
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            # Сохранение
            res_image = Image.fromarray(res_rgb)
            res_image.save(output_path)
            
        print(f"Обработано и сохранено: {filename}")

    print(f"\nИнференс YOLO завершен. Результаты в: {output_dir}")

if __name__ == '__main__':
    run_yolo_inference()
