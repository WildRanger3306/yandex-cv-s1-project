import torch
import os
import glob
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
import matplotlib.pyplot as plt

def run_fcos_inference():
    # 1. Настройка путей
    config_file = 'configs/fcos/fcos_minecraft.py'
    checkpoint_dir = 'artifacts/fcos'
    test_images_dir = 'datasets/minecraft/images/test'
    output_dir = 'artifacts/inference/fcos'
    
    os.makedirs(output_dir, exist_ok=True)

    # 2. Поиск лучшего чекпоинта
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'best_*.pth'))
    if not checkpoints:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, 'epoch_*.pth'))
    
    if not checkpoints:
        print(f"Ошибка: Чекпоинты не найдены в {checkpoint_dir}")
        return
    
    # Берем самый свежий/лучший
    checkpoint_file = max(checkpoints, key=os.path.getmtime)
    print(f"Использование чекпоинта: {checkpoint_file}")

    # 3. Инициализация модели
    register_all_modules()
    # Используем torch для проверки доступности GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = init_detector(config_file, checkpoint_file, device=device)
    
    # Инициализация визуализатора
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # 4. Прогон по тестовым изображениям
    image_paths = glob.glob(os.path.join(test_images_dir, '*.jpg'))
    print(f"Найдено {len(image_paths)} изображений для инференса.")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        
        # Инференс
        result = inference_detector(model, img_path)
        
        # Чтение изображения
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        
        # Визуализация
        # pred_score_thr - вот здесь меняется порог уверенности для отрисовки BBOX
        # Мы ставим 0.15, чтобы увидеть даже не очень уверенные предсказания
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            wait_time=0,
            out_file=output_path,
            pred_score_thr=0.15 
        )
        print(f"Обработано: {filename}")

    print(f"\nИнференс FCOS завершен. Результаты в: {output_dir}")

if __name__ == '__main__':
    run_fcos_inference()
