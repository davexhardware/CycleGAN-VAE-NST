import shutil
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

image_labels=['282','51493','82975','85655','090327',
              '076921','066850','059070', '064218','048722',
              '046466']

def merge_results(src_dir,dest_dir,epochs):
    for epoch in epochs:
        source_dir=src_dir.format(epoch=epoch)
        dest_dir=root_dest_dir.format(epoch=epoch)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
        images=os.listdir(source_dir)
        for image in images:
            for label in image_labels:
                if label in image:
                    shutil.copy(source_dir+image,dest_dir+image)
                    break

def create_comparison(export_root, compare_dir, exclusion_keys=None):
        
    if exclusion_keys is None:
        exclusion_keys = list()
    model_dirs = [
        d for d in os.listdir(export_root) if os.path.isdir(os.path.join(export_root, d)) and all([key not in d for key in exclusion_keys])
    ]
    model_dirs.sort(key=lambda x: x.split('_')[1]+x.split('_')[0] if '_' in x else x)

    # Use one model as reference for real images.
    ref_model = model_dirs[0]
    ref_path = os.path.join(export_root, ref_model)
    real_images = [
        f for f in os.listdir(ref_path) if '_real' in f
    ]

    # Collect paths and labels in two lists.
    for real_name in real_images:
        images = [''.join([ref_path,'/',real_name]) ]
        captions = ["Real"]
        size_px=256
        # Find matching fake in each model directory.
        for model in model_dirs:
            fake_name = real_name.replace("_real", "_fake")
            fake_path = ''.join([export_root, model,'/', fake_name])
            if os.path.exists(fake_path):
                images.append(fake_path)
                captions.append(model)
        fig, axes = plt.subplots(1, len(images), figsize=(size_px*len(images)*3/size_px, size_px*3/size_px))
        for ax, path, label in zip(axes, images, captions):
            img =Image.open(path)
            ax.imshow(img)
            ax.set_title(label, color="black")
            ax.axis("off")

        fig.tight_layout()
        out_name = real_name.replace("_real", "_comparison")
        plt.savefig(os.path.join(compare_dir, out_name), bbox_inches="tight")
        plt.close(fig)


#### Comment this section if you don't want 
#### to load the images to the export_results folder
root_src_dir='./results/real2op{epoch}/test_latest/images/'
epochs=[200,250,300,350]
root_dest_dir='./results/export_results/real2op{epoch}/'
merge_results(root_src_dir,root_dest_dir,epochs)
root_src_dir='./results/real2op{epoch}_reference_idt/test_latest/images/'
epochs=[100,150,200]
root_dest_dir='./results/export_results/real2op{epoch}_reference_idt/'
merge_results(root_src_dir,root_dest_dir,epochs)

#### Comment this section if you don't want
#### to create the comparison images
root_export='./results/export_results/'
comparison_dir = root_export+"comparison_s/"
os.makedirs(comparison_dir, exist_ok=True)
exclude_dir_keys=['comparison','real2op150', 'real2op250', 'real2op350']
create_comparison(root_export,comparison_dir,exclude_dir_keys)