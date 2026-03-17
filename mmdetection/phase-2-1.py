import json
import os

def check_json_structure(file_path):
    print(f"Checking {file_path}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        required_keys = ['images', 'annotations', 'categories']
        missing_keys = [key for key in required_keys if key not in data]
        
        if not missing_keys:
            print(f"  [OK] JSON structure is correct.")
            print(f"  - Images: {len(data['images'])}")
            print(f"  - Annotations: {len(data['annotations'])}")
            print(f"  - Categories: {len(data['categories'])}")
        else:
            print(f"  [ERROR] Missing keys: {missing_keys}")
            
    except Exception as e:
        print(f"  [ERROR] Failed to load JSON: {e}")

def main():
    # Пытаемся найти папку annotations относительно текущей директории
    base_path = 'datasets/minecraft/annotations'

    files = ['train.json', 'valid.json', 'test.json']
    
    for filename in files:
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path):
            check_json_structure(file_path)
        else:
            print(f"[WARNING] File not found: {file_path}")

if __name__ == "__main__":
    main()
