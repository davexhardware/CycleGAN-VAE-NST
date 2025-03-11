import re
import shutil
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

image_labels=['282','51493','82975','85655','090327',
              '076921','066850','059070', '064218','048722',
              '046466']

def merge_results(src_dir,format_dest_dir,model_base,model_identifier,epochs):
    for epoch in epochs:
        source_dir=src_dir.format(epoch=epoch,mb=model_base,mi=model_identifier)
        dest_dir=format_dest_dir.format(epoch=epoch,mb=model_base,mi=model_identifier)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
        images=os.listdir(source_dir)
        for image in images:
            for label in image_labels:
                if label in image:
                    shutil.copy(source_dir+image,dest_dir+image)
                    break

def create_comparison(export_root, compare_dir, inclusion_keys=None,exclusion_keys=None):
        
    if inclusion_keys is None:
        inclusion_keys = list()
    if exclusion_keys is None:
        exclusion_keys = list()
    model_dirs = [
        d for d in os.listdir(export_root) if os.path.isdir(os.path.join(export_root, d)) and any([k in d for k in inclusion_keys]) and all([k not in d for k in exclusion_keys])
    ]
    model_dirs.sort(key=lambda x: x.split('_')[1]+re.findall(r'(\d){2,3}',x)[0] if '_' in x else re.findall(r'(\d){2,3}',x)[0])

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
model_base='real2op'
epochs=['060','080']
model_identifier='_vaegan_complex'
root_src_dir="./results/{mb}{epoch}{mi}/test_latest/images/"
root_dest_dir='./results/export_results/{mb}{epoch}{mi}/'
merge_results(root_src_dir,root_dest_dir,model_base,model_identifier,epochs)
"""
model_base='real2op'
epochs=[200,250,300,350]
model_identifier='_vaegan_complex'
root_src_dir="./results/{mb}{epoch}{mi}/test_latest/images/"
root_dest_dir='./results/export_results/{mb}{epoch}{mi}/'
merge_results(root_src_dir,root_dest_dir,model_base,model_identifier,epochs)
"""
#### Comment this section if you don't want
#### to create the comparison images
root_export='./results/export_results/'
comparison_dir = root_export+"comparison_vaegan/"
os.makedirs(comparison_dir, exist_ok=True)
include_dir_keys=['vaegan']
exclude_dir_keys=['comparison']
create_comparison(root_export,comparison_dir,include_dir_keys,exclude_dir_keys)