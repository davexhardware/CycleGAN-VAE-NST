from unaligned_dataset import UnalignedDataset
from options import train_options
from torch import save,load,serialization, Tensor
from torchvision.tv_tensors import Image
from torchvision.transforms import functional as F
from matplotlib import pyplot as plt
import numpy as np
import os
train_pt_floatsize='trainA_pt_16'
if not os.path.isdir('./datasets/'+train_pt_floatsize):
    os.mkdir('./datasets/'+train_pt_floatsize)

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
        ts=load(path)
        print(f"Shape: {ts.shape}, Min: {ts.min().item()}, Max: {ts.max().item()}")
        pts.append(ts)
        if i==5:
            break
show(pts)
 