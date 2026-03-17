import cv2
import os
import glob
import torch
import mmcv
from ultralytics import YOLO
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS

def process_video_yolo(model_path, input_path, output_path):
    print(f"Начало обработки видео YOLO: {input_path}")
    model = YOLO(model_path)
    
    # Открываем видео для получения свойств
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Настройка записи
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Инференс в режиме стрима для экономии памяти
    results = model.predict(source=input_path, stream=True, conf=0.25)
    
    count = 0
    for r in results:
        frame_plotted = r.plot() # Возвращает BGR
        out.write(frame_plotted)
        count += 1
        if count % 100 == 0:
            print(f"YOLO: Обработано {count}/{total_frames} кадров")
    
    out.release()
    print(f"Видео YOLO сохранено в: {output_path}")

def process_video_fcos(config_path, checkpoint_path, input_path, output_path):
    print(f"Начало обработки видео FCOS: {input_path}")
    register_all_modules()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = init_detector(config_path, checkpoint_path, device=device)
    
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    video_reader = mmcv.VideoReader(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, video_reader.fps, (video_reader.width, video_reader.height))

    count = 0
    for frame in video_reader:
        # Инференс
        result = inference_detector(model, frame)
        
        # Визуализация на кадре
        visualizer.add_datasample(
            'result',
            frame,
            data_sample=result,
            draw_gt=False,
            wait_time=0,
            pred_score_thr=0.15 # Порог из этапа 5-1
        )
        
        # Получаем изображение с отрисованными боксами (Visualizer работает в RGB)
        frame_vis = visualizer.get_image()
        frame_vis_bgr = cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR)
        
        video_writer.write(frame_vis_bgr)
        
        count += 1
        if count % 100 == 0:
            print(f"FCOS: Обработано {count}/{len(video_reader)} кадров")
    
    video_writer.release()
    print(f"Видео FCOS сохранено в: {output_path}")

if __name__ == '__main__':
    video_input = 'datasets/minecraft/video.mp4'
    video_dir = 'artifacts/videos'
    os.makedirs(video_dir, exist_ok=True)

    # Пути к моделям
    yolo_best = 'artifacts/yolo/weights/best.pt'
    fcos_config = 'configs/fcos/fcos_minecraft.py'
    
    # Поиск лучшего чекпоинта FCOS
    fcos_checkpoints = glob.glob('artifacts/fcos/best_*.pth')
    if not fcos_checkpoints:
        fcos_checkpoints = glob.glob('artifacts/fcos/epoch_*.pth')
    
    if os.path.exists(video_input):
        # Обработка YOLO
        if os.path.exists(yolo_best):
            process_video_yolo(yolo_best, video_input, os.path.join(video_dir, 'yolo_inference.mp4'))
        else:
            print("Ошибка: Веса YOLO не найдены.")

        # Обработка FCOS
        if fcos_checkpoints and os.path.exists(fcos_config):
            fcos_best = max(fcos_checkpoints, key=os.path.getmtime)
            process_video_fcos(fcos_config, fcos_best, video_input, os.path.join(video_dir, 'fcos_inference.mp4'))
        else:
            print("Ошибка: Чекпоинт или конфиг FCOS не найдены.")
    else:
        print(f"Ошибка: Входное видео {video_input} не найдено.")
