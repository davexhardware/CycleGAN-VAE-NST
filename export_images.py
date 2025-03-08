import shutil
import os

image_labels=['282','51493','82975','85655']
root_src_dir='./results/real2op{epoch}/test_latest/images/'
epochs=[200,250,300,350]
root_dest_dir='./results/export_results/real2op{epoch}/'
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
