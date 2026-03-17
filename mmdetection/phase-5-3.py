import os
import glob
import random
import matplotlib.pyplot as plt
from PIL import Image

def visualize_inference_comparison():
    fcos_dir = 'artifacts/inference/fcos'
    yolo_dir = 'artifacts/inference/yolo'
    output_path = 'artifacts/metrics/inference_comparison_paired.png'

    # Получаем списки имен файлов (без путей)
    fcos_files = {os.path.basename(f) for f in glob.glob(os.path.join(fcos_dir, '*.jpg'))}
    yolo_files = {os.path.basename(f) for f in glob.glob(os.path.join(yolo_dir, '*.jpg'))}

    # Находим общие изображения для сравнения
    common_files = list(fcos_files.intersection(yolo_files))
    
    if not common_files:
        print("Ошибка: Нет общих изображений в папках FCOS и YOLO для сравнения.")
        return

    # Выбираем 5 случайных изображений (или меньше, если всего меньше 5)
    num_to_show = min(len(common_files), 5)
    selected_files = random.sample(common_files, num_to_show)
    
    print(f"Визуализация сравнения для {num_to_show} пар изображений.")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Сравнение инференса на одних и тех же данных: FCOS (верх) vs YOLO (низ)', fontsize=16)

    for i in range(5):
        if i < num_to_show:
            filename = selected_files[i]
            
            # Пути к изображениям
            fcos_img_path = os.path.join(fcos_dir, filename)
            yolo_img_path = os.path.join(yolo_dir, filename)
            
            # Отрисовка FCOS
            img_fcos = Image.open(fcos_img_path)
            axes[0, i].imshow(img_fcos)
            axes[0, i].set_title(f"FCOS: {filename[:10]}...")
            
            # Отрисовка YOLO
            img_yolo = Image.open(yolo_img_path)
            axes[1, i].imshow(img_yolo)
            axes[1, i].set_title(f"YOLO: {filename[:10]}...")
        
        # Скрываем оси
        axes[0, i].axis('off')
        axes[1, i].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Парная визуализация сохранена в: {output_path}")
    plt.show()

if __name__ == '__main__':
    visualize_inference_comparison()
