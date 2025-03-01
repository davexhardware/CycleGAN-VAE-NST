import kagglehub
import os
import shutil
import random
from tqdm import tqdm

full_dataset_path = "./full"
destination_folder_path = './trainA'
ds_size = 10000

print("Download path set to ", os.path.abspath(full_dataset_path))
print("Destination folder set to ", os.path.abspath(destination_folder_path))

# Download latest version only if the dataset is not already downloaded
if not os.path.exists(full_dataset_path):
    downloaded_path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    print("Downloaded path: ", downloaded_path)

    # Move the downloaded dataset to the full_dataset_path
    if os.path.exists(downloaded_path):
        shutil.move(downloaded_path, full_dataset_path)
        print("Dataset downloaded successfully.")

# Extract the dataset
if os.path.exists(full_dataset_path) and not os.path.exists(destination_folder_path):
    print(f"Extracting {ds_size} randomly selected images to {destination_folder_path}")

    celeba_imgs_path = os.path.join(full_dataset_path, "img_align_celeba/img_align_celeba")

    os.makedirs(destination_folder_path, exist_ok=True)

    images = [f for f in os.listdir(celeba_imgs_path) if os.path.isfile(os.path.join(celeba_imgs_path, f))]
    random.shuffle(images)
    images = images[:ds_size]  # Use only 10k images for training

    for image in tqdm(images):
        shutil.copy(os.path.join(celeba_imgs_path, image), os.path.join(destination_folder_path, image))

print("Done.")
