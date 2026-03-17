import json
import os
import matplotlib.pyplot as plt
import glob
import re

def visualize_fcos_metrics():
    # 1. Попытка найти scalars.json
    log_files_json = glob.glob('artifacts/fcos/**/scalars.json', recursive=True)
    
    epochs, iterations, loss, mAP, val_epochs = [], [], [], [], []

    if log_files_json:
        log_path = max(log_files_json, key=os.path.getmtime)
        print(f"Загрузка JSON логов из: {log_path}")
        with open(log_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'loss' in data and 'step' in data:
                    iterations.append(data['step'])
                    loss.append(data['loss'])
                if 'coco/bbox_mAP' in data:
                    mAP.append(data['coco/bbox_mAP'])
                    val_epochs.append(data.get('epoch', len(val_epochs)+1))
    else:
        # 2. Если JSON нет, парсим текстовый .log
        log_files_txt = glob.glob('artifacts/fcos/**/*.log', recursive=True)
        if not log_files_txt:
            print("Ошибка: Лог-файлы не найдены в artifacts/fcos.")
            return
        
        log_path = max(log_files_txt, key=os.path.getmtime)
        print(f"JSON не найден. Парсинг текстового лога: {log_path}")
        
        with open(log_path, 'r') as f:
            for line in f:
                # Ищем строку с loss: Epoch(train) [1][ 50/1154] ... loss: 3.8606
                loss_match = re.search(r'loss:\s+([0-9.]+)', line)
                step_match = re.search(r'\[\s*(\d+)/', line)
                epoch_match = re.search(r'Epoch\(train\)\s+\[(\d+)\]', line)
                
                if loss_match and step_match and epoch_match:
                    loss.append(float(loss_match.group(1)))
                    # Примерный расчет итерации для графика
                    loss_epoch = int(epoch_match.group(1))
                    loss_step = int(step_match.group(1))
                    iterations.append((loss_epoch - 1) * 1000 + loss_step) 

                # Ищем строку с mAP: coco/bbox_mAP: 0.1234
                map_match = re.search(r'coco/bbox_mAP:\s+([0-9.]+)', line)
                if map_match:
                    mAP.append(float(map_match.group(1)))
                    val_epochs.append(len(mAP))

    if not loss and not mAP:
        print("Ошибка: Не удалось извлечь данные из логов.")
        return

    # Создание графиков
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(iterations, loss, label='Train Loss', color='blue')
    plt.xlabel('Iter (approx)')
    plt.ylabel('Loss')
    plt.title('FCOS Training Loss')
    plt.grid(True)
    plt.legend()

    if mAP:
        plt.subplot(1, 2, 2)
        plt.plot(val_epochs, mAP, marker='o', label='mAP (0.5:0.95)', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.title('FCOS Validation mAP')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    output_dir = 'artifacts/metrics'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fcos_training_metrics.png')
    plt.savefig(output_path)
    print(f"Графики сохранены в: {output_path}")
    plt.show()

if __name__ == '__main__':
    visualize_fcos_metrics()
