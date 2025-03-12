import os
import shutil

model_name="real2op200"
src_dir=f'./datasets/{model_name}/test_latest/images'
eval_dir=f'./datasets/eval/{model_name}'
real_subdir=f'{eval_dir}/gt'
fake_subdir=f'{eval_dir}/gene'
real_label='_real'
fake_label='_fake'

if not os.path.exists(real_subdir):
    os.makedirs(real_subdir)
if not os.path.exists(fake_subdir):
    os.makedirs(fake_subdir)

for file in os.listdir(src_dir):
    if file.endswith(real_label+'.png'):
        shutil.copy(os.path.join(src_dir, file), os.path.join(real_subdir, file.replace(real_label,'')))
    elif file.endswith(fake_label+'.png'):
        shutil.copy(os.path.join(src_dir, file), os.path.join(fake_subdir, file.replace(fake_label,'')))