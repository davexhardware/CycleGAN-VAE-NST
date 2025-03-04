from data.custom_dataset_nst.scripts import holdout, download_from_kaggle

ds_dir = 'C:\\Users\\DavideSavoia\\Documents\\DeepProj\\CycleGAN-VAE-NST\\celeba\\celeba'
working_dir = './celeba'


def download_dataset(path):
    download_from_kaggle(
        path=path,
        ds_name='jessicali9530/celeba-dataset'
    )
download_dataset(ds_dir)