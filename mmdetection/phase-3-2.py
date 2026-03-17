import os
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS

def test_fcos_pretrained():
    # 3-2-2 Импорт конфига и чекпоинта
    config_file = 'configs/fcos/fcos_minecraft.py'
    checkpoint_file = 'checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'
    
    # Путь к тестовому изображению
    img_path = 'datasets/minecraft/images/test/grass_desert_-_cow_3_1434_jpg.rf.110c73a314ed050344613c069fbc1328.jpg'
    output_path = 'artifacts/inference/test_pretrained.jpg'
    
    # Проверка наличия файлов
    if not os.path.exists(config_file):
        print(f"Error: Config file not found at {config_file}")
        return
    if not os.path.exists(checkpoint_file):
        print(f"Error: Checkpoint file not found at {checkpoint_file}")
        return
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}")
        return

    # Регистрация модулей и инициализация модели
    register_all_modules()
    print(f"Initializing model with config: {config_file}")
    model = init_detector(config_file, checkpoint_file, device='cpu')
    
    # 3-2-3 Запуск инференса
    print(f"Running inference on: {img_path}")
    result = inference_detector(model, img_path)
    
    # 3-2-4 Визуализация и сохранение результата
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    img = mmcv.imread(img_path)
    img = mmcv.imconvert(img, 'bgr', 'rgb')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        wait_time=0,
        out_file=output_path
    )
    print(f"Result saved to: {output_path}")

if __name__ == "__main__":
    test_fcos_pretrained()
