import shutil
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_labels=['282','51493','82975','85655','090327',
              '076921','066850','059070', '064218','048722',
              '046466']

def merge_results(src_dir,dest_dir,epochs):
    for epoch in epochs:
        source_dir=root_src_dir.format(epoch=epoch)
        dest_dir=root_dest_dir.format(epoch=epoch)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
        images=os.listdir(source_dir)
        for image in images:
            for label in image_labels:
                if label in image:
                    shutil.copy(source_dir+image,dest_dir+image)
                    break


root_src_dir='./results/real2op{epoch}/test_latest/images/'
epochs=[200,250,300,350]
root_dest_dir='./results/export_results/real2op{epoch}/'
merge_results(root_src_dir,root_dest_dir,epochs)
root_src_dir='./results/real2op{epoch}_reference_idt/test_latest/images/'
epochs=[100,150,200]
root_dest_dir='./results/export_results/real2op{epoch}_reference_idt/'
merge_results(root_src_dir,root_dest_dir,epochs)

root_export='./results/export_results/'

model_dirs = [
    d for d in os.listdir(root_export) if os.path.isdir(os.path.join(root_export, d))
]
model_dirs.sort(key=lambda x: x.split('_')[1]+x.split('_')[0] if '_' in x else x)

compare_dir = root_export+"comparisons/"
os.makedirs(compare_dir, exist_ok=True)

# Use one model as reference for real images.
ref_model = model_dirs[0]
ref_path = os.path.join(root_export, ref_model)
real_images = [
    f for f in os.listdir(ref_path) if '_real' in f
]

try:
    font = ImageFont.load_default()
except:
    font = None
   

    # Collect paths and labels in two lists.
    
for real_name in real_images:
    # Open the real image.
    real_img = Image.open(os.path.join(ref_path, real_name)).convert("RGB")
    images = [real_img]
    captions = ["Real"]

    # Find matching fake in each model directory.
    for model in model_dirs:
        fake_name = real_name.replace("_real", "_fake")
        fake_path = os.path.join(root_export, model, fake_name)
        if os.path.exists(fake_path):
            images.append(fake_path)
            captions.append(model)

    # Combine them horizontally with captions.
    #total_width = sum(im.width for im in images)
    #max_height = max(im.height for im in images)
    #combined = Image.new("RGB", (total_width, max_height), "black")
    #x_offset = 0
    fig, axes = plt.subplots(1, len(images))
    for ax, path, label in zip(axes, images, image_labels):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(label, color="white")
        ax.axis("off")

    fig.tight_layout()
    out_name = real_name.replace("_real", "_comparison")
    plt.savefig(os.path.join(compare_dir, out_name), bbox_inches="tight")
    plt.close(fig)
    """
    for im, cap in zip(images, captions):
        draw = ImageDraw.Draw(im)
        draw.text((5, 5), cap, font=font, fill="white")
        combined.paste(im, (x_offset, 0))
        x_offset += im.width

    # Save the comparison image.
    out_name = real_name.replace("_real", "_comparison")
    combined.save(os.path.join(compare_dir, out_name))
"""