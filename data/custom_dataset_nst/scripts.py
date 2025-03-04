import os
import random
import shutil

import kagglehub
from tqdm import tqdm

def download_from_kaggle(ds_name, path):
    print("Download path set to ", os.path.abspath(path))
    if not os.path.exists(path):
        downloaded_path = kagglehub.dataset_download(ds_name)
        print("Downloaded path: ", downloaded_path)

        # Move the downloaded dataset to the given path
        if os.path.exists(downloaded_path):
            shutil.move(downloaded_path, path)
            print("Dataset downloaded successfully.")

def holdout(src_path, dest_path, label='A', train_size=10000, test_size=1000):
    print("Destination folder set to ", os.path.abspath(dest_path))

    dest_train = dest_path + '/train' + label
    dest_test = dest_path + '/test' + label

    if os.path.exists(src_path):
        print(f"Generating train_{label} and test_{label} images from {os.path.abspath(dest_path)}")

        images = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
        random.shuffle(images)
        train_images = images[:train_size]  # Use train_size images for training
        test_images = images[train_size:train_size + test_size]  # Use test_size images for testing

        if not os.path.exists(dest_train):
            os.makedirs(dest_train, exist_ok=True)

            print(f"Adding {len(train_images)} train images to {os.path.abspath(dest_path)}")
            for image in tqdm(train_images):
                shutil.copy(os.path.join(src_path, image), os.path.join(dest_train, image))
        else:
            print(f'Folder {dest_train} already exists. Skipped.')

        if not os.path.exists(dest_test):
            os.makedirs(dest_test, exist_ok=True)

            print(f"Adding {len(test_images)} test images to {os.path.abspath(dest_path)}")
            for image in tqdm(test_images):
                shutil.copy(os.path.join(src_path, image), os.path.join(dest_test, image))
        else:
            print(f'Folder {dest_test} already exists. Skipped.')