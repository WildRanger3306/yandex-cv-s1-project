# Этап 4-1. Обучение модели FCOS на датасете Minecraft
import os
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules

def train_fcos():
    # Регистрация всех модулей MMDetection
    register_all_modules()

    config_path = 'configs/fcos/fcos_minecraft.py'
    work_dir = 'artifacts/fcos'

    # Создаем директорию для артефактов, если она не существует
    # Это гарантирует, что логи, веса (.pth) и результаты валидации 
    # будут сохранены именно в artifacts/fcos
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    if not os.path.exists(config_path):
        print(f"Ошибка: Конфигурационный файл {config_path} не найден.")
        return

    print(f"Загрузка конфигурации из {config_path}...")
    cfg = Config.fromfile(config_path)
    
    # Указываем рабочую директорию. MMEngine Runner автоматически сохранит туда:
    # 1. Логи обучения (scalars.json, log.txt)
    # 2. Веса модели (best_bbox_mAP.pth, epoch_12.pth и т.д.)
    # 3. Контрольные результаты (визуализации в папке vis_data)
    cfg.work_dir = work_dir
    
    # Убеждаемся, что логирование и визуализатор включены
    if 'default_hooks' in cfg:
        cfg.default_hooks.logger = dict(type='LoggerHook', interval=50)
        cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=1, save_best='coco/bbox_mAP')
        cfg.default_hooks.visualization = dict(type='DetVisualizationHook', draw=True)
        
    # Явно настраиваем Visualizer для создания scalars.json
    cfg.visualizer = dict(
        type='DetLocalVisualizer',
        vis_backends=[dict(type='LocalVisBackend')],
        name='visualizer',
        save_dir=work_dir  
    )

    print(f"Инициализация Runner. Все результаты будут в: {os.path.abspath(work_dir)}")
    runner = Runner.from_cfg(cfg)
    
    # Запуск обучения
    runner.train()
    
    print(f"Обучение FCOS завершено. Проверьте {work_dir} для получения логов и весов.")

if __name__ == '__main__':
    train_fcos()
