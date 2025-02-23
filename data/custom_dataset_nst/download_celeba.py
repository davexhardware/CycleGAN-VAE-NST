import kagglehub
import os
import shutil
import random

# Download latest version
#path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
# Extract the dataset in a certain quantity of portraits
path="C:\\Users\\DavideSavoia\\OneDrive - Politecnico di Bari\\1M\\DeepL\\progetto\\Portraits dataset\\img_align_celeba"
print("Path to dataset files:", path)
if os.path.exists(path):
    print("Dataset downloaded successfully.")
    destination_folder = 'datasets/trainA_portraits'
    os.makedirs(destination_folder, exist_ok=True)

    images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    random.shuffle(images)
    images = images[:10000]

    for image in images:
        shutil.copy(os.path.join(path, image), os.path.join(destination_folder, image))

    print(f"Copied {len(images)} images to {destination_folder}.")
#Cut up to 10k pictures in folder
"""destination_folder = 'datasets/trainA_portraits'
images = [f for f in os.listdir(destination_folder) if os.path.isfile(os.path.join(destination_folder, f))]
images = images[:10000]
for image in [f for f in os.listdir(destination_folder) if os.path.isfile(os.path.join(destination_folder, f))]:
    if image not in images:
        os.remove(os.path.join(destination_folder, image))"""