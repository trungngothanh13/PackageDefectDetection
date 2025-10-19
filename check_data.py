"""
Check classification dataset quality and balance
"""

import os
from collections import Counter
from PIL import Image

def check_dataset_balance():
    """Check if classes are balanced"""
    
    print("=" * 70)
    print("CLASSIFICATION DATASET ANALYSIS")
    print("=" * 70)
    
    splits = ['train', 'val', 'test']
    classes = ['good', 'ok', 'bad']
    
    for split in splits:
        split_path = f"data/classification/{split}"
        
        if not os.path.exists(split_path):
            print(f"\n{split.upper()}: Not found")
            continue
        
        print(f"\n{split.upper()} SET:")
        print("-" * 70)
        
        class_counts = {}
        total = 0
        
        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                count = len(images)
                class_counts[class_name] = count
                total += count
                
                # Check for very small images
                small_images = 0
                corrupted_images = 0
                
                for img_file in images[:10]:  # Sample first 10
                    try:
                        img_path = os.path.join(class_path, img_file)
                        img = Image.open(img_path)
                        width, height = img.size
                        if width < 100 or height < 100:
                            small_images += 1
                    except:
                        corrupted_images += 1
                
                print(f"  {class_name:8s}: {count:4d} images", end="")
                
                if small_images > 0:
                    print(f"{small_images} very small images detected", end="")
                if corrupted_images > 0:
                    print(f"{corrupted_images} corrupted files!", end="")
                print()
        
        print(f"  {'TOTAL':8s}: {total:4d} images")
        
        # Check balance
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else 0
            
            if imbalance_ratio > 3:
                print(f"\nWARNING: Class imbalance detected!")
                print(f"Balance your classes (collect more of minority class)")
            elif imbalance_ratio > 2:
                print(f"\nModerately imbalance: {imbalance_ratio:.1f}:1")
            else:
                print(f"\nClasses are well balanced ({imbalance_ratio:.1f}:1)")
    
    # Get train counts
    train_path = "data/classification/train"
    if os.path.exists(train_path):
        train_counts = {}
        for class_name in classes:
            class_path = os.path.join(train_path, class_name)
            if os.path.exists(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                train_counts[class_name] = count
        
        total_train = sum(train_counts.values())
        
        if total_train < 300:
            print("\nCRITICAL: Very few training images!")
            print(f"   Current: {total_train} images")
            print(f"   Recommended: 500+ images per class (1500+ total)")
            print("   Action: Collect more images!")
        elif total_train < 500:
            print(f"\nLow training data: {total_train} images")
            print("   Recommended: 500+ images per class")
            print("   Current model might underfit")
        else:
            print(f"\nGood amount of training data: {total_train} images")
        
        # Check individual classes
        print("\nPer-class recommendations:")
        for class_name, count in train_counts.items():
            if count < 150:
                print(f"{class_name}: {count} images - collect 50-100 more")
            elif count < 200:
                print(f"{class_name}: {count} images - could use 30-50 more")
            else:
                print(f"{class_name}: {count} images - sufficient")

def check_mislabeled_images():    
    print("\n" + "=" * 70)
    print("MISLABELING CHECK")
    print("=" * 70)
    
    print("\nManual check recommended:")
    print("  1. Look at 'good' folder - any bad/ok images mixed in?")
    print("  2. Look at 'ok' folder - is it clearly different from good/bad?")
    print("  3. Look at 'bad' folder - any good/ok images mixed in?")
    print("\nEven 5-10% mislabeled images can cause accuracy to plateau!")
    
    print("\n" + "=" * 70)


def main():
    check_dataset_balance()
    check_mislabeled_images()


if __name__ == "__main__":
    main()