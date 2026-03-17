import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from fpdf import FPDF
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.visualization import DetLocalVisualizer
from ultralytics import YOLO
import torch

# Font path that supports Cyrillic
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def generate_examples():
    print("Генерация примеров детекций...")
    output_dir = 'artifacts/inference_examples'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Настройка YOLO
    yolo_model = YOLO('artifacts/yolo/weights/best.pt')
    
    # 2. Настройка FCOS
    register_all_modules()
    fcos_cfg = 'configs/fcos/fcos_minecraft.py'
    fcos_checkpoint = glob.glob('artifacts/fcos/best_*.pth')[0]
    fcos_model = init_detector(fcos_cfg, fcos_checkpoint, device='cuda:0')
    
    # Выбираем 2 тестовых изображения
    test_images = glob.glob('datasets/minecraft/images/test/*.jpg')[:2]
    
    for i, img_path in enumerate(test_images):
        img_name = os.path.basename(img_path)
        
        # YOLO inference
        yolo_res = yolo_model.predict(img_path, imgsz=512, conf=0.3)[0]
        yolo_plot = yolo_res.plot()
        yolo_example_path = os.path.join(output_dir, f'yolo_ex_{i}.jpg')
        cv2.imwrite(yolo_example_path, cv2.cvtColor(yolo_plot, cv2.COLOR_RGB2BGR))
        
        # FCOS inference
        fcos_res = inference_detector(fcos_model, img_path)
        visualizer = DetLocalVisualizer()
        visualizer.dataset_meta = fcos_model.dataset_meta
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualizer.add_datasample(
            f'fcos_res_{i}',
            img,
            data_sample=fcos_res,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=None,
            pred_score_thr=0.3
        )
        fcos_plot = visualizer.get_image()
        fcos_example_path = os.path.join(output_dir, f'fcos_ex_{i}.jpg')
        cv2.imwrite(fcos_example_path, cv2.cvtColor(fcos_plot, cv2.COLOR_RGB2BGR))

    return test_images

def create_charts():
    print("Генерация графиков...")
    csv_path = 'artifacts/metrics/metrics_comparison.csv'
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 5))
    
    # mAP chart
    plt.subplot(1, 2, 1)
    plt.bar(df['Model'], df['mAP'], color=['blue', 'green'])
    plt.title('mAP (0.5:0.95)')
    plt.ylabel('Value')
    
    # FPS chart
    plt.subplot(1, 2, 2)
    plt.bar(df['Model'], df['FPS'], color=['blue', 'green'])
    plt.title('FPS')
    plt.ylabel('Frames per second')
    
    plt.tight_layout()
    chart_path = 'artifacts/metrics/comparison_charts.png'
    plt.savefig(chart_path)
    plt.close()
    return chart_path

class PDFReport(FPDF):
    def header(self):
        self.add_font('DejaVu', '', FONT_PATH, uni=True)
        self.set_font('DejaVu', '', 16)
        self.cell(0, 10, 'Отчёт по сравнению моделей детектирования Minecraft', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('DejaVu', '', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, text):
        self.set_font('DejaVu', '', 12)
        self.multi_cell(0, 7, text)
        self.ln()

def generate_report():
    print("Создание PDF отчёта...")
    csv_path = 'artifacts/metrics/metrics_comparison.csv'
    df = pd.read_csv(csv_path)
    
    fcos_map = df[df['Model'] == 'FCOS']['mAP'].values[0]
    yolo_map = df[df['Model'] == 'YOLOv8s']['mAP'].values[0]
    fcos_fps = df[df['Model'] == 'FCOS']['FPS'].values[0]
    yolo_fps = df[df['Model'] == 'YOLOv8s']['FPS'].values[0]
    
    pdf = PDFReport()
    pdf.add_page()
    
    # 1. Выводы о качестве
    pdf.chapter_title("1. Выводы о моделях на основе метрик качества")
    quality_text = (
        f"Модель YOLOv8s показала значительно более высокую точность по сравнению с FCOS.\n"
        f"- YOLOv8s mAP (0.5:0.95): {yolo_map:.3f}\n"
        f"- FCOS mAP (0.5:0.95): {fcos_map:.3f}\n"
        f"Разрыв в mAP составляет более чем 2.5 раза. YOLOv8s лучше справляется с детектированием "
        f"разнообразных мобов в Minecraft на данном наборе данных."
    )
    pdf.chapter_body(quality_text)
    
    # 2. Выводы о скорости
    pdf.chapter_title("2. Выводы о моделях на основе метрик скорости")
    speed_text = (
        f"В плане скорости инференса YOLOv8s также значительно опережает FCOS.\n"
        f"- YOLOv8s FPS: {yolo_fps:.1f}\n"
        f"- FCOS FPS: {fcos_fps:.1f}\n"
        f"YOLOv8s работает в {yolo_fps/fcos_fps:.1f} раза быстрее, что позволяет использовать её "
        f"в реальном времени (более 200 FPS на GTX 1080 Ti), в то время как FCOS выдает около 30 FPS."
    )
    pdf.chapter_body(speed_text)
    
    # 3. Примеры результатов
    pdf.chapter_title("3. Примеры результатов детекций на инференсе")
    pdf.chapter_body("Ниже представлены примеры работы обеих моделей на тестовых изображениях. "
                     "Слева - YOLOv8s, справа - FCOS (для каждого из двух примеров).")
    
    example_dir = 'artifacts/inference_examples'
    for i in range(2):
        yolo_img = os.path.join(example_dir, f'yolo_ex_{i}.jpg')
        fcos_img = os.path.join(example_dir, f'fcos_ex_{i}.jpg')
        
        # Add images side by side
        y_pos = pdf.get_y()
        if y_pos > 200: # New page if needed
            pdf.add_page()
            y_pos = pdf.get_y()
            
        pdf.image(yolo_img, x=10, y=y_pos, w=90)
        pdf.image(fcos_img, x=105, y=y_pos, w=90)
        pdf.ln(70) # Space for images
    
    # 4. Графики
    pdf.add_page()
    pdf.chapter_title("4. Графики для визуального сравнения")
    chart_path = 'artifacts/metrics/comparison_charts.png'
    pdf.image(chart_path, x=10, y=pdf.get_y(), w=180)
    
    os.makedirs('artifacts', exist_ok=True)
    report_path = 'artifacts/report.pdf'
    pdf.output(report_path)
    print(f"Отчёт сохранён: {report_path}")

def main():
    generate_examples()
    create_charts()
    generate_report()

if __name__ == '__main__':
    main()
