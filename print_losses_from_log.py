from options.train_options import TrainOptions
from util.visualizer import Visualizer
import os
import re
opt = TrainOptions().parse()
visualizer= Visualizer(opt)
dataset_size=1000
src_losses= os.path.join(opt.checkpoints_dir, opt.name,'loss_log.txt')
print(src_losses)
"""
    Usage: 
    First start visdom with: python -m visdom.server
    Then: python print_losses_from_log.py --name portraits2op_reskvae --dataroot ./ --gpu_ids -1
    Open browser and go to http://localhost:8097
 """
with open(src_losses,'r') as f:
    rows = f.readlines()
    losses=[]
    for i,lossrow in enumerate(rows):
        if('Training Loss') not in lossrow:
            row = lossrow.strip().replace(')', '')
            parts = re.findall(r'(\w+): ([\d.]+)', row)
            parsed = {k: float(v) for k, v in parts}
            epoch=parsed.pop('epoch')
            iters=parsed.pop('iters')
            _=parsed.pop('time')
            _=parsed.pop('data')
            losses.append(parsed)
            if(i%opt.print_freq==0):
                visualizer.plot_current_losses(
                    epoch,
                    float(iters)/dataset_size,
                    parsed
                )
