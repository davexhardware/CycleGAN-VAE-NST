from unaligned_dataset import UnalignedDataset
from options import train_options
from torch import save,load,serialization
from torchvision.tv_tensors import Image
from torchvision.transforms import functional as F
from matplotlib import pyplot as plt
import numpy as np
import os
train_pt_floatsize='trainA_pt_16_unnormalized'
if not os.path.isdir('./datasets/'+train_pt_floatsize):
    os.mkdir('./datasets/'+train_pt_floatsize)
# Usage: call the function, will move all images without 'inverted' 
# (or any other filtering feature) in the name from src_dir to dst_dir
"""def gather_images(src_dir, dst_dir, filter_func=None):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for root, _, fnames in os.walk(src_dir):
        for fname in fnames:
            if filter_func and not filter_func(fname):
                continue
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(dst_dir, fname)
            os.rename(src_path, dst_path)

def filter_images_inverted(fname):
    return 'inverted' not in fname

src_dir = './Data'
dst_dir = './op_dataset_merged'
gather_images(src_dir, dst_dir, filter_images_inverted)"""

### Usage: python data\iterate_dataset.py --preprocess resize_and_crop --dataroot .\datasets\ --gpu_ids -1 --load_size 286
opt = train_options.TrainOptions().parse()
dataset = UnalignedDataset(opt)
for el in range(dataset.__len__()):
    item=dataset.__getitem__(el)
    path_A=item['A_paths'].replace('trainA',train_pt_floatsize)+'.pt'
    save(item['A'], path_A)
    

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fix.show()
    input()
    
serialization.add_safe_globals([Image])
dir='./datasets/'+train_pt_floatsize
if not os.path.isdir(dir):
    print('No such directory')
    exit()
i=0
pts=[]
for root, _, fnames in sorted(os.walk(dir)):
    for fname in fnames:
        i+=1
        path = os.path.join(root, fname)
        print(path)
        pts.append(load(path))
        if i==5:
            break
show(pts)
 