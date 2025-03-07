from data.unaligned_dataset import UnalignedDataset
from options import train_options
from torch import save,load,serialization, Tensor
from torchvision.tv_tensors import Image
from torchvision.transforms import functional as F
from matplotlib import pyplot as plt
import numpy as np
import os

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
                

""" Copy the image files from the folders in <src_root>/[<folders>] **temporarly** inside the 
 <dataroot>/trainA folder, then stores the images as tensor files in <dataroot>/<folder>_pt.
 In order to use the tensors for training the model, you should replace the newly created folders
 with trainA and trainB in the <dataroot> directory."""

### Usage: python iterate_dataset.py --transform_float16 --preprocess resize_and_crop --dataroot .\datasets\ --gpu_ids -1 --load_size <LOAD_SIZE> --crop_size <CROP>

opt = train_options.TrainOptions().parse()
dataset_dir=opt.dataroot
src_root='./downloads/'
folders=['img_align_celeba','onepiece']

if not os.path.exists(dataset_dir+'trainB'):
    os.mkdir(dataset_dir+'trainB')
    blank_image = Tensor(np.zeros((3, opt.crop_size, opt.crop_size), dtype=np.float32))
    blank_image_pil = F.to_pil_image(blank_image)
    blank_image_pil.save(os.path.join(dataset_dir+'trainB', 'blank_image.png'))

for folder in folders:
    if os.path.exists(src_root+folder):
        src_dir=dataset_dir+'trainA'
        # COMMENT IF YOU DON'T WANT TO DELETE THE CONTENTS OF THE FOLDER 'trainA' EACH TIME
        if os.path.exists(src_dir):
            for root, dirs, files in os.walk(src_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
                os.rmdir(src_dir)
        # STOP COMMENTING
        if not os.path.exists(src_dir):
            src_dir = os.path.join(dataset_dir, 'trainA')
            os.makedirs(src_dir, exist_ok=True)
            root_dir = os.path.join(src_root, folder)
            for root, _, files in os.walk(root_dir):
                for file in files:
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(src_dir, file)
                    with open(src_file, 'rb') as fsrc, open(dest_file, 'wb') as fdst:
                        fdst.write(fsrc.read())
        else:
            print(f'Folder {src_dir} exists')
        dest_dir=dataset_dir+folder+'_pt'
        if(not os.path.exists(dest_dir)):
            os.mkdir(dest_dir)
        dataset = UnalignedDataset(opt)
        for el in range(max(dataset.__len__(),20000)):
            item=dataset.__getitem__(el)
            path_A=item['A_paths'].replace(src_dir,dest_dir)+'.pt'
            save(item['A'], path_A)
            """serialization.add_safe_globals([Image])
            dir= dest_dir
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
        """
    else:
        print(f'No such directory: {src_root+folder}')
        exit()

for folder in folders:
    pt_folder=dataset_dir+folder+'_pt'
    if os.path.exists(pt_folder):
        print(f'Folder {pt_folder} exists')
    else:
        print(f'No such directory: {pt_folder}')
        exit()

print('Done. Tensors of the images have been saved in the respective folders.')
print('In order to train the model, substitute the trainA and trainB folders with the tensors of the images you want to use as domain A and B.')
exit()