import os
from pathlib import Path
from data.custom_dataset_nst.scripts import holdout

# TODO: set the path to the downloaded dataset
ds_dir_original = Path('C:\\Users\\vitod\\Datasets\\onepiece\\Data\\Data')
# TODO: set the path to the directory where the flattened dataset will be stored
ds_dir_flat = Path('C:\\Users\\vitod\\Datasets\\onepiece\\Flatten_data')
# TODO: set the path to the working directory
working_dir = Path('../onepiece')


# Usage: call the function, will move all images without 'inverted'
# (or any other filtering feature) in the name from src_dir to dst_dir
def flatten_dataset(src_dir, dst_dir, filter_func=None):
    if not os.path.exists(ds_dir_flat):
        ds_dir_flat.mkdir(parents=True, exist_ok=True)
        for root, _, fnames in os.walk(src_dir):
            for fname in fnames:
                if filter_func and not filter_func(fname):
                    continue
                character_name = root.split('\\')[-1]
                src_path = os.path.join(root, fname)
                dst_path = os.path.join(dst_dir, f'{character_name}_{fname}')
                os.rename(src_path, dst_path)
    print("Flatten complete.")


def filter_images_inverted(file_name):
    return 'inverted' not in file_name


# if not os.path.exists(working_dir) and os.path.exists(ds_dir_flat):
#     working_dir.mkdir(parents=True, exist_ok=True)
#
#     # Iterate over all files in the source directory and its subdirectories
#     for file in ds_dir_flat.iterdir():
#         # Check if the file is an image
#         if file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
#             with Image.open(file) as img:
#                 print(f"Img mode: {img.mode}")
#                 if img.mode == 'RGB' and all(channel == img.split()[0] for channel in img.split()):
#                     print(f"Image {file.name} is in grayscale but stored in RGB format.")
#                 # Copy the file to the destination directory
#                 else:
#                     shutil.copy(file, working_dir / file.name)

# Create the destination directory if doesn't exist
flatten_dataset(ds_dir_original, ds_dir_flat, filter_images_inverted)
