from ultralytics import YOLO
import cv2
from PIL import Image
import os

# 3-4. Проверка инференса на pretrained-модели YOLOv8s
def test_yolo_pretrained():
    model_path = 'yolov8s.pt'
    # Используем то же тестовое изображение, что и в предыдущих тестах
    test_img = 'datasets/minecraft/images/test/160_png_jpg.rf.3fafd0f2c05721d89ec2b6e382cb89e3.jpg'
    output_dir = 'artifacts/inference/yolo_val/'
    
    if not os.path.exists(model_path):
        print(f"Ошибка: Файл {model_path} не найден.")
        return

    print(f"Загрузка модели {model_path}...")
    model = YOLO(model_path)
    
    print(f"Запуск инференса для {test_img}...")
    results = model.predict(test_img, save=False)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for result in results:
        # Отрисовка результатов
        res_plotted = result.plot()
        # Конвертация BGR (OpenCV) в RGB (PIL)
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Сохранение результата
        filename = os.path.basename(test_img)
        output_path = os.path.join(output_dir, filename)
        
        res_image = Image.fromarray(res_rgb)
        res_image.save(output_path)
        print(f"Результат успешно сохранен в: {output_path}")

if __name__ == '__main__':
    test_yolo_pretrained()
