from unaligned_dataset import UnalignedDataset
from options import train_options
from torch import save,load,serialization
from torchvision.tv_tensors import Image
from torchvision.transforms import functional as F
from matplotlib import pyplot as plt
import numpy as np
import os
"""opt = train_options.TrainOptions().parse()
dataset = UnalignedDataset(opt)
for el in range(dataset.__len__()):
    item=dataset.__getitem__(el)
    path_A=item['A_paths'].replace('trainA','trainA_pt_32')+'.pt'
    save(item['A'], path_A)
    """
    
   
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
dir='./datasets/trainA_pt_32'
if not os.path.isdir(dir):
    print('No such directory')
    exit()
for root, _, fnames in sorted(os.walk(dir)):
    for fname in fnames:
        path = os.path.join(root, fname)
        print(path)
        pt=load(path)
        show(pt)
        break
 