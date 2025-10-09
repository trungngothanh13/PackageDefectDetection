"""
Split dataset into train/val/test sets for YOLO training
"""

import os
import shutil
import random
from pathlib import Path


# Configuration
RAW_IMAGES_DIR = "data/raw/images"
RAW_LABELS_DIR = "data/raw/labels"

TRAIN_IMAGES_DIR = "data/dataset/train/images"
TRAIN_LABELS_DIR = "data/dataset/train/labels"
VAL_IMAGES_DIR = "data/dataset/val/images"
VAL_LABELS_DIR = "data/dataset/val/labels"
TEST_IMAGES_DIR = "data/dataset/test/images"
TEST_LABELS_DIR = "data/dataset/test/labels"

# Split ratios
TRAIN_RATIO = 0.7  # 70%
VAL_RATIO = 0.2    # 20%
TEST_RATIO = 0.1   # 10%

# Random seed for reproducibility
RANDOM_SEED = 42


def create_directories():
    """Create train/val/test directories if they don't exist"""
    directories = [
        TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR,
        VAL_IMAGES_DIR, VAL_LABELS_DIR,
        TEST_IMAGES_DIR, TEST_LABELS_DIR
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created/verified: {directory}")


def get_image_label_pairs():
    """Get list of (image_path, label_path) pairs"""
    pairs = []
    
    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"❌ Error: {RAW_IMAGES_DIR} not found!")
        return pairs
    
    # Get all image files
    image_files = [f for f in os.listdir(RAW_IMAGES_DIR) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in image_files:
        image_path = os.path.join(RAW_IMAGES_DIR, image_file)
        
        # Get corresponding label file
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(RAW_LABELS_DIR, label_file)
        
        # Only include pairs where both image and label exist
        if os.path.exists(label_path):
            pairs.append((image_path, label_path))
        else:
            print(f"⚠️  Warning: Label not found for {image_file}")
    
    return pairs


def split_data(pairs):
    """Split data into train/val/test sets"""
    # Shuffle with fixed seed
    random.seed(RANDOM_SEED)
    random.shuffle(pairs)
    
    total = len(pairs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]
    
    return train_pairs, val_pairs, test_pairs


def copy_files(pairs, images_dest, labels_dest, split_name):
    """Copy image and label files to destination directories"""
    print(f"\n📦 Copying {split_name} set ({len(pairs)} samples)...")
    
    for image_path, label_path in pairs:
        # Copy image
        image_filename = os.path.basename(image_path)
        shutil.copy2(image_path, os.path.join(images_dest, image_filename))
        
        # Copy label
        label_filename = os.path.basename(label_path)
        shutil.copy2(label_path, os.path.join(labels_dest, label_filename))
    
    print(f"✓ Copied {len(pairs)} images and labels to {split_name}")


def verify_split():
    """Verify the split results"""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    train_images = len(os.listdir(TRAIN_IMAGES_DIR))
    train_labels = len(os.listdir(TRAIN_LABELS_DIR))
    val_images = len(os.listdir(VAL_IMAGES_DIR))
    val_labels = len(os.listdir(VAL_LABELS_DIR))
    test_images = len(os.listdir(TEST_IMAGES_DIR))
    test_labels = len(os.listdir(TEST_LABELS_DIR))
    
    total = train_images + val_images + test_images
    
    print(f"\n📊 Split Results:")
    print(f"   Train: {train_images} images, {train_labels} labels ({train_images/total*100:.1f}%)")
    print(f"   Val:   {val_images} images, {val_labels} labels ({val_images/total*100:.1f}%)")
    print(f"   Test:  {test_images} images, {test_labels} labels ({test_images/total*100:.1f}%)")
    print(f"   Total: {total} samples")
    
    # Check for mismatches
    if train_images != train_labels:
        print(f"⚠️  Warning: Train images/labels mismatch!")
    if val_images != val_labels:
        print(f"⚠️  Warning: Val images/labels mismatch!")
    if test_images != test_labels:
        print(f"⚠️  Warning: Test images/labels mismatch!")
    
    if (train_images == train_labels and 
        val_images == val_labels and 
        test_images == test_labels):
        print("\n✅ All sets have matching image/label counts!")


def main():
    print("=" * 60)
    print("YOLO Dataset Splitter")
    print("=" * 60)
    
    # Step 1: Create directories
    print("\n1️⃣  Creating directories...")
    create_directories()
    
    # Step 2: Get image-label pairs
    print("\n2️⃣  Scanning raw data...")
    pairs = get_image_label_pairs()
    
    if not pairs:
        print("❌ No valid image-label pairs found!")
        return
    
    print(f"✓ Found {len(pairs)} valid image-label pairs")
    
    # Step 3: Split data
    print("\n3️⃣  Splitting data...")
    train_pairs, val_pairs, test_pairs = split_data(pairs)
    print(f"   Train: {len(train_pairs)} samples")
    print(f"   Val:   {len(val_pairs)} samples")
    print(f"   Test:  {len(test_pairs)} samples")
    
    # Step 4: Copy files
    print("\n4️⃣  Copying files...")
    copy_files(train_pairs, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, "train")
    copy_files(val_pairs, VAL_IMAGES_DIR, VAL_LABELS_DIR, "val")
    copy_files(test_pairs, TEST_IMAGES_DIR, TEST_LABELS_DIR, "test")
    
    # Step 5: Verify
    verify_split()
    
    print("\n" + "=" * 60)
    print("✅ Dataset split complete!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   1. Review the split ratios above")
    print("   2. Create dataset.yaml configuration")
    print("   3. Start training!")


if __name__ == "__main__":
    main()