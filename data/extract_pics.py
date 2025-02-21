import os
import shutil
from pathlib import Path
from PIL import Image

source_dir = Path('dataset')
destination_dir = Path('dataset_rgb')

# Create the destination directory if doesn't exist
if(os.path.exists(destination_dir)):
    shutil.rmtree(destination_dir)
destination_dir.mkdir(parents=True, exist_ok=True)
# Usage: call the function, will move all images without 'inverted' 
# (or any other filtering feature) in the name from src_dir to dst_dir
"""def gather_images(src_dir, dst_dir, filter_func=None):
    for root, _, fnames in os.walk(src_dir):
        for fname in fnames:
            if filter_func and not filter_func(fname):
                continue
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(dst_dir, fname)
            os.rename(src_path, dst_path)

def filter_images_inverted(fname):
    return 'inverted' not in fname

gather_images(source_dir, destination_dir, filter_images_inverted)"""

# Iterate over all files in the source directory and its subdirectories
for file in source_dir.iterdir():
    # Check if the file is an image
    if file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        with Image.open(file) as img:
            print(f"Img mode: {img.mode}")
            if img.mode == 'RGB' and all(channel == img.split()[0] for channel in img.split()):
                print(f"Image {file.name} is in grayscale but stored in RGB format.")
            # Copy the file to the destination directory
            else:
                shutil.copy(file, destination_dir / file.name)

print("Extraction complete.")