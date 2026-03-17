import json
import os
import matplotlib.pyplot as plt
import pandas as pd

def analyze_class_distribution(json_path, output_image_path):
    print(f"Analyzing class distribution for {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Mapping category IDs to names
        categories = {cat['id']: cat['name'] for cat in data['categories']}
        
        # Count occurrences of each category_id in annotations
        class_counts = {}
        for ann in data['annotations']:
            class_id = ann['category_id']
            class_name = categories.get(class_id, f"Unknown({class_id})")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
        df = df.sort_values(by='Count', ascending=False)
        
        # Visualization
        plt.figure(figsize=(12, 8))
        plt.barh(df['Class'], df['Count'], color='skyblue')
        plt.xlabel('Count')
        plt.ylabel('Class Name')
        plt.title(f'Class Distribution in {os.path.basename(json_path)}')
        plt.gca().invert_yaxis()  # Most frequent at the top
        plt.tight_layout()
        
        # Save visualization
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        plt.savefig(output_image_path)
        print(f"  [OK] Distribution plot saved to {output_image_path}")
        
        # Print summary
        print("\nClass summary:")
        print(df.to_string(index=False))
        
        # Basic imbalance check
        max_count = df['Count'].max()
        min_count = df['Count'].min()
        ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\nImbalance Analysis:")
        print(f"  - Max count: {max_count} ({df['Class'].iloc[0]})")
        print(f"  - Min count: {min_count} ({df['Class'].iloc[-1]})")
        print(f"  - Ratio (max/min): {ratio:.2f}")
        
        if ratio > 10:
            print("  [CONCLUSION] Significant class imbalance detected.")
        elif ratio > 3:
            print("  [CONCLUSION] Moderate class imbalance detected.")
        else:
            print("  [CONCLUSION] Class distribution is relatively balanced.")
            
    except Exception as e:
        print(f"  [ERROR] Failed to analyze {json_path}: {e}")

def main():
    train_json = 'datasets/minecraft/annotations/train.json'
    output_img = 'artifacts/metrics/class_distribution.png'
    
    if os.path.exists(train_json):
        analyze_class_distribution(train_json, output_img)
    else:
        print(f"[WARNING] File not found: {train_json}")

if __name__ == "__main__":
    main()
