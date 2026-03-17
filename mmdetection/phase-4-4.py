import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_yolo_metrics():
    results_path = 'artifacts/yolo/results.csv'
    output_dir = 'artifacts/metrics'
    
    if not os.path.exists(results_path):
        print(f"Ошибка: Файл {results_path} не найден. Сначала запустите обучение YOLO.")
        return

    # Загрузка данных
    df = pd.read_csv(results_path)
    # Удаляем лишние пробелы в названиях колонок (бывает в YOLOv8)
    df.columns = [c.strip() for c in df.columns]

    # Создание графиков
    plt.figure(figsize=(15, 10))

    # 1. Графики Loss (Box, Class)
    plt.subplot(2, 2, 1)
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
    plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('YOLO Box Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss')
    plt.plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('YOLO Class Loss')
    plt.legend()
    plt.grid(True)

    # 2. Метрики точности (mAP50, mAP50-95)
    plt.subplot(2, 2, 3)
    plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('YOLO mAP@50')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', color='darkgreen')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('YOLO mAP@50-95')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # Сохранение результата
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'yolo_training_metrics.png')
    plt.savefig(output_path)
    print(f"Графики метрик YOLO сохранены в: {output_path}")
    plt.show()

if __name__ == '__main__':
    visualize_yolo_metrics()
