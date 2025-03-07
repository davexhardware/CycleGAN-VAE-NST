from data.custom_dataset_nst.scripts import download_from_kaggle

ds_dir = 'C:\\Users\\DavideSavoia\\Documents\\DeepProj\\CycleGAN-VAE-NST\\celeba\\celeba'
working_dir = './celeba'

download_from_kaggle(
    path=ds_dir,
    ds_name='jessicali9530/celeba-dataset'
)
