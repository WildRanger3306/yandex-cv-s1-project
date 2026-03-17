import json
import os
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize_test_example(json_path, img_dir, output_path):
    print(f"Visualizing one test example from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Mapping category IDs to names
        categories = {cat['id']: cat['name'] for cat in data['categories']}
        
        # Pick a random image from the dataset
        img_info = random.choice(data['images'])
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_path = os.path.join(img_dir, file_name)
        
        if not os.path.exists(img_path):
            print(f"  [ERROR] Image not found: {img_path}")
            return

        # Get annotations for this image
        anns = [ann for ann in data['annotations'] if ann['image_id'] == img_id]
        
        # Load image
        img = Image.open(img_path)
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)
        
        # Draw bounding boxes
        for ann in anns:
            # COCO format is [x_min, y_min, width, height]
            bbox = ann['bbox']
            class_id = ann['category_id']
            class_name = categories.get(class_id, f"Unknown({class_id})")
            
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3], 
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            plt.text(
                bbox[0], bbox[1] - 5, class_name, 
                color='white', weight='bold', 
                bbox=dict(facecolor='red', alpha=0.5)
            )
        
        plt.title(f"Example: {file_name} (ID: {img_id})")
        plt.axis('off')
        plt.tight_layout()
        
        # Save visualization
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"  [OK] Visualization saved to {output_path}")
            
    except Exception as e:
        print(f"  [ERROR] Failed to visualize example: {e}")

def main():
    test_json = 'datasets/minecraft/annotations/test.json'
    test_img_dir = 'datasets/minecraft/images/test'
    output_img = 'artifacts/inference/eda_example.jpg'
    
    if os.path.exists(test_json):
        visualize_test_example(test_json, test_img_dir, output_img)
    else:
        print(f"[WARNING] File not found: {test_json}")

if __name__ == "__main__":
    main()
