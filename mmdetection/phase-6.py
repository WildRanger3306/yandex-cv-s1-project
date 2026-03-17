import os
import time
import torch
import pandas as pd
import glob
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
from ultralytics import YOLO

def get_fcos_metrics():
    print("Оценка FCOS...")
    register_all_modules()
    config_path = 'configs/fcos/fcos_minecraft.py'
    checkpoint_dir = 'artifacts/fcos'
    
    # Поиск лучшего чекпоинта
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'best_*.pth'))
    if not checkpoints:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, 'epoch_*.pth'))
    if not checkpoints:
        print("Чекпоинты FCOS не найдены.")
        return None
    
    checkpoint_path = max(checkpoints, key=os.path.getmtime)
    print(f"Использование чекпоинта FCOS: {checkpoint_path}")

    cfg = Config.fromfile(config_path)
    cfg.load_from = checkpoint_path
    cfg.work_dir = 'artifacts/fcos_val'
    
    # Настраиваем test_dataloader на тестовый набор
    cfg.test_dataloader.dataset.ann_file = 'annotations/test.json'
    cfg.test_dataloader.dataset.data_prefix = dict(img='images/test/')
    cfg.test_evaluator.ann_file = cfg.data_root + 'annotations/test.json'

    runner = Runner.from_cfg(cfg)
    
    # Замер времени для FPS
    start_time = time.time()
    metrics = runner.test()
    end_time = time.time()
    
    num_images = len(runner.test_dataloader.dataset)
    total_time = end_time - start_time
    fps = num_images / total_time
    
    return {
        'Model': 'FCOS',
        'mAP': metrics.get('coco/bbox_mAP', 0),
        'mAP_50': metrics.get('coco/bbox_mAP_50', 0),
        'FPS': fps
    }

def get_yolo_metrics():
    print("Оценка YOLOv8...")
    model_path = 'artifacts/yolo/weights/best.pt'
    if not os.path.exists(model_path):
        print("Веса YOLO не найдены.")
        return None
    
    model = YOLO(model_path)
    
    # Замер времени для FPS
    # split='test' возьмет данные из секции 'test' в data.yaml
    results = model.val(data='datasets/minecraft/data.yaml', split='test', imgsz=512, plots=False)
    
    # У YOLO результаты в results.results_dict
    # maps: mAP50-95, mAP50, mAP75...
    metrics_dict = results.results_dict
    
    # Расчет FPS на основе внутреннего замера YOLO (ms на изображение)
    total_speed_ms = results.speed['preprocess'] + results.speed['inference'] + results.speed['postprocess']
    fps = 1000.0 / total_speed_ms
    
    return {
        'Model': 'YOLOv8s',
        'mAP': metrics_dict.get('metrics/mAP50-95(B)', 0),
        'mAP_50': metrics_dict.get('metrics/mAP50(B)', 0),
        'FPS': fps
    }

def main():
    results = []
    
    # Получаем метрики для обеих моделей
    fcos_res = get_fcos_metrics()
    if fcos_res: results.append(fcos_res)
    
    yolo_res = get_yolo_metrics()
    if yolo_res: results.append(yolo_res)
    
    if not results:
        print("Нет данных для сравнения.")
        return

    # Создание таблицы
    df = pd.DataFrame(results)
    
    # Вывод в консоль
    print("\n--- Сравнение метрик на тестовом наборе ---")
    print(df.to_string(index=False))
    
    # Сохранение в CSV
    output_dir = 'artifacts/metrics'
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'metrics_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nРезультаты сохранены в: {csv_path}")

if __name__ == '__main__':
    main()
