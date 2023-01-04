from pytorch_gan_metrics import (
    get_inception_score,
    get_fid_from_directory,
    get_inception_score_and_fid_from_directory,get_fid)
from pytorch_gan_metrics import ImageDataset
import torch.utils.data as data_utils
import torchvision.transforms as transforms


def calc_fid_score(path, dataset):
    # IS, IS_std = get_inception_score_from_directory(
    #   'path/to/images')
    print("path to images " + path)
    print("path to statistics " + dataset + '_statistics.npz')
    fid = get_fid_from_directory(path, dataset + '_statistics.npz')
    # (IS, IS_std), FID = get_inception_score_and_fid_from_directory(
    #   'path/to/images', 'path/to/statistics.npz')

    return fid

def calc_fid_score_grid(path, dataset):
    if dataset == 'fashion-mnist' or dataset == 'mnist':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    elif dataset == 'cifar':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    imset = ImageDataset(path, exts=['png', 'jpg'])
    loader = data_utils.DataLoader(imset, batch_size=50, num_workers=4)
    print(dataset)
    if (dataset == "cifar"):
        fid = get_fid(loader, 'cifar10.train.npz')
    else:
        fid = get_fid(loader, dataset + '_statistics.npz')
    # (IS, IS_std), FID = get_inception_score_and_fid_from_directory(
    #   'path/to/images', 'path/to/statistics.npz')

    return fid

def calc_IS_score_from_grid(images):
    IS, IS_std = get_inception_score(
        images)
    return IS


def calc_IS_score(path, dataset):
    if dataset == 'fashion-mnist' or dataset == 'mnist':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    elif dataset == 'cifar':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    imgset = ImageDataset(path, exts=['png', 'jpg'])
    loader = data_utils.DataLoader(imgset, batch_size=50, num_workers=4)
    IS, IS_std = get_inception_score(loader)
    #IS, IS_std = get_inception_score_from_directory(path)
    print("path to images " + path)
    # (IS, IS_std), FID = get_inception_score_and_fid_from_directory(
    #   'path/to/images', 'path/to/statistics.npz')

    return IS
