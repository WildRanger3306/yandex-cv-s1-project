import json
import os

def check_image_annotation_consistency(file_path, image_dir):
    print(f"Checking consistency for {file_path}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        images_in_json = {img['id']: img['file_name'] for img in data['images']}
        annotations_image_ids = {ann['image_id'] for ann in data['annotations']}
        
        # 1. Check if all annotations point to existing image IDs in JSON
        orphans = [ann_id for ann_id in annotations_image_ids if ann_id not in images_in_json]
        
        # 2. Check if all images in JSON exist on disk
        missing_on_disk = []
        for img_id, file_name in images_in_json.items():
            full_path = os.path.join(image_dir, file_name)
            if not os.path.exists(full_path):
                missing_on_disk.append(file_name)
        
        print(f"  - Total images in JSON: {len(images_in_json)}")
        print(f"  - Total unique images referenced in annotations: {len(annotations_image_ids)}")
        
        if not orphans:
            print(f"  [OK] All annotations refer to valid image IDs.")
        else:
            print(f"  [ERROR] Found {len(orphans)} annotations pointing to non-existent image IDs.")
            
        if not missing_on_disk:
            print(f"  [OK] All images listed in JSON exist in {image_dir}.")
        else:
            print(f"  [ERROR] {len(missing_on_disk)} images are missing on disk.")
            
    except Exception as e:
        print(f"  [ERROR] Failed to process {file_path}: {e}")

def main():
    base_path = 'datasets/minecraft/annotations'
    img_base_path = 'datasets/minecraft/images' # Corrected path
    
    # Mapping JSON files to their image directories
    file_map = {
        'train.json': 'train',
        'valid.json': 'valid',
        'test.json': 'test'
    }
    
    for json_file, img_dir_suffix in file_map.items():
        file_path = os.path.join(base_path, json_file)
        image_dir = os.path.join(img_base_path, img_dir_suffix)
        
        if os.path.exists(file_path):
            check_image_annotation_consistency(file_path, image_dir)
        else:
            print(f"[WARNING] File not found: {file_path}")

if __name__ == "__main__":
    main()
