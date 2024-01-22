import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as io

import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader

from argparse import Namespace
import plenoptic as po
from einops import rearrange
import warnings

from ppm.planted import sample_translations, sample_rotations
from ppm.dataset import DAVIS

from ppm.baselines import C, MC, F, Spyr, L, LNL
from ppm.models import PP, mPP, QP, mQP
from ppm.networks import CNN, Unet

home = os.path.expanduser("~")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_checkpoint(name, verbose=True):
    """
    verbose: boolean
        print args, performance and extra information
            (num_params, train/test time)
    """
    logs_path = home + '/PolarPrediction/checkpoints' + name
    args = load_args(logs_path, verbose=verbose)
    train_metrics, test_metrics = load_performance(logs_path, verbose=verbose)
    model = load_model(logs_path, args)
    if verbose:
        print_info(logs_path)
    return model, train_metrics, test_metrics

def get_dataloaders(batch_size, name='DAVIS', subset=0):
    """ presets
    """
    if name in ['translate', 'rotate', 'translate_rotate', 'translate_open']:
        kwargs = {
            'dataset': 'PLANTED',
            'transform': name,
            'filter_size': 16,
            'normalize': True,
            'gray': True,
            'image_size': 128,
            'num_downs': 1,
            'num_crops': 1,
            'fold': 2017,
            'subset': subset,
            'overfit': False,
            'mini_batch': batch_size,
        }
    elif name == 'VANH':
        kwargs = {
            'dataset': 'VANH',
            'normalize': True,
            'gray': True,
            'overfit': False,
            'mini_batch': batch_size,
        }
    elif name == 'DAVIS':
        kwargs = {
            'dataset': 'DAVIS',
            'normalize': True,
            'gray': True,
            'image_size': 128,
            'num_downs': 1,
            'num_crops': 1,
            'fold': 2017,
            'subset': subset,
            'overfit': False,
            'mini_batch': batch_size,
        }
    args = Namespace(**kwargs)
    train_dataloader, test_dataloader, data_stats = init_data(args)
    return train_dataloader, test_dataloader, data_stats

def init_data(args):
    if 'PLANTED' in args.dataset:
        assert args.gray == 1, 'PLANTED is grayscale only'
        print('Loading PLANTED dataset')
        N = 384 #100 #
        S = 128 #int(256 // (2 ** args.num_downs))
        # condition
        if 'noise' in args.dataset:
            print("training on noise")
            X = torch.tensor(
                np.stack([po.tools.synthetic_images.pink_noise(S, 1.)
                          for b in range(N)], axis=0)[:, None, None]
            ).float()
        else:
            # sample N random frames from natural video
            X = torch.empty((N, 1, 1, S, S))
            data_path = home + '/Documents/datasets/DAVIS'
            dataset = DAVIS(
                data_path, split='train', normalize=args.normalize,
                gray=args.gray, image_size=128, clip_length=11,
                n_levels=args.num_downs, n_crops=args.num_crops, fold=args.fold,
                subset=args.subset,
            )
            idx = np.random.choice(np.arange(len(dataset)), N, replace=False)
            for i, j in enumerate(idx):
                X[i] = dataset[j][:, 0:1]
            del dataset

        f = args.filter_size
        T = 11
        if args.transform == 'translate':
            X = sample_translations(X, f=f, T=T, vmax=f//4)
        elif args.transform == 'rotate':
            X = sample_rotations(X, f=f, T=T, amax=360//4)
        elif args.transform == 'translate_open':
            X = sample_translations(X, f=f, T=T, vmax=f//4, strict=False)
        elif args.transform == 'translate_rotate':
            Tr = sample_translations(X, f=f, T=T, vmax=f//4)
            Ro = sample_rotations(X, f=f, T=T, amax=360//4) 
            X = torch.cat((Tr, Ro)) # twice as large

        X = po.tools.rescale(X)
        if args.normalize:
            data_stats = {'mean': X.mean(), 'std': X.std()}
            X = standardize(X, data_stats)
        else:
            # leave data in range [0-1]
            data_stats = {'mean': torch.zeros(1), 'std': torch.ones(1)}

        idx = np.random.permutation(len(X))
        train_dataset = X[idx[:-len(X)//10]]
        test_dataset = X[idx[-len(X)//10:]]
        # print(train_dataset.shape, test_dataset.shape)
        collate_fn = None

    elif args.dataset == 'VANH':
        assert args.gray == 1, 'VANH is grayscale only'
        print('Loading VANH dataset')
        data_path = home + "/Documents/datasets/vid075-chunks/"

        # loading to torch floats
        movie = io.loadmat(data_path + 'movie.mat')['movie']
        movie = np.transpose(movie, (3, 2, 0, 1)) # chunk, time, hight, width
        movie = movie.reshape(56 * 64, 128, 128)
        movie = torch.from_numpy(movie).float()
        
        # reshaping
        movie = movie[:, 16:, :] # crop missing top part of image
        # TODO double size of dataset by flipping left-right (removing bias)
        movie = movie[:-(len(movie)%11)] # drop_last=True
        movie = rearrange(movie, '(B T) H W -> B 1 T H W', T=11)

        # scaling
        movie = po.tools.rescale(movie)
        if args.normalize:
            data_stats = {'mean': movie.mean(), 'std': movie.std()}
            movie = standardize(movie, data_stats)
        else:
            # leave the data in [0-1]
            data_stats = {'mean': torch.zeros(1), 'std': torch.ones(1)}

        # train-test split
        # idx = np.random.permutation(len(movie))
        # train_idx = idx[:-len(movie)//10]
        # test_idx = idx[-len(movie)//10:]
        # precomputed with seed 0
        test_idx = np.array(
            [288, 294, 147, 285, 300, 177,  99, 197, 243, 115, 265,  72,  25,
             165, 311, 174, 313,  39, 193,  88,  70,  87, 292, 242, 277, 211,
             9, 195, 251, 192, 117,  47, 172]
        )
        train_idx = np.delete(np.arange(325), test_idx)
        train_dataset = movie[train_idx]
        test_dataset = movie[test_idx]

        collate_fn = None

    elif args.dataset == 'DAVIS':
        print('Loading DAVIS dataset')
        data_path = home + '/Documents/datasets/DAVIS'
        train_dataset = DAVIS(
            data_path, split='train', normalize=args.normalize,  gray=args.gray,
            image_size=args.image_size, clip_length=11, n_levels=args.num_downs,
            n_crops=args.num_crops, fold=args.fold, subset=args.subset,
            overfit=args.overfit
        )
        test_dataset = DAVIS(
            data_path, split='val', normalize=args.normalize,  gray=args.gray,
            image_size=args.image_size, clip_length=11, n_levels=args.num_downs,
            n_crops=args.num_crops, fold=args.fold, subset=args.subset,
        )
        data_stats = {'mean': train_dataset.mean, 'std': train_dataset.std}
        collate_fn = None

    else:
        print("incorrect dataset argument")

    if args.overfit:
        # select first examples to form one mini_batch
        train_dataset = Subset(train_dataset, list(range(args.mini_batch)))

    # TODO set num_workers per dataset
    # num_workers = 1
    num_workers = torch.get_num_threads()

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.mini_batch, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.mini_batch, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn
    )
    return train_dataloader, test_dataloader, data_stats

def load_args(logs_path, verbose=False):
    args = json.load(open(logs_path + "/args.json"))
    args = Namespace(**args)
    if verbose:
        print(vars(args))
    return args

def load_performance(logs_path, verbose=False):
    assert os.path.isdir(logs_path + '/loss__test'), 'Didnt compute test loss'
    train_metrics = {}
    test_metrics = {}

    df = tflog2pandas(logs_path + '/loss__test')
    idx = df['value'].argmin()
    test_metrics["mse"] = df['value'].iloc[idx]
    test_metrics["psnr"] = tflog2pandas(
        logs_path + '/psnr__test')['value'].iloc[idx]
    test_metrics["ssim"] = tflog2pandas(
        logs_path + '/ssim__test')['value'].iloc[idx]
    train_metrics["mse"] = tflog2pandas(
        logs_path + '/loss__train')['value'].iloc[idx]
    train_metrics["psnr"] = tflog2pandas(
        logs_path + '/psnr__train')['value'].iloc[idx]
    train_metrics["ssim"] = tflog2pandas(
        logs_path + '/ssim__train')['value'].iloc[idx]

    if verbose:    
        print("\n lowest test loss at epoch ", int(df['step'].iloc[idx]))
        print('train: ', train_metrics)
        print('test: ', test_metrics)
    return train_metrics, test_metrics

def tflog2pandas(path):
    # modified from following github repo:
    # theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator
    )
    import traceback
    runlog_data = pd.DataFrame({
        "metric": [],
        "value": [],
        "step": [],
        "wall_time": []
    })
    try:
        event_acc = EventAccumulator(path, size_guidance={"scalars": 0})
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            wall_time = list(map(lambda x: x.wall_time, event_list))
            r = {
                "metric": [tag] * len(step),
                "value": values,
                "step": step,
                "wall_time": wall_time
            }
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

def load_model(logs_path, args, best_train=False):
    """
    best_train: boolean
        load model weights that minimize train error
    """
    model = init_model(args)
    model = model.to(device)
    if args.model not in ['C', 'Spyr', 'cMC']:
        if best_train:
            saved_model = logs_path + "/model_train.pt"
        else:
            saved_model = logs_path + "/model.pt"
        if 'cuda' in str(device):
            model.load_state_dict(torch.load(saved_model))
        else:
            model.load_state_dict(
                torch.load(saved_model, map_location=torch.device('cpu'))
            )
    model.eval()
    return model

def print_info(logs_path):
    try:
        with open(logs_path + '/info.txt', 'r') as text_file:
            print(text_file.read())
    except FileNotFoundError:
        print(f'no info.txt file in {logs_path}')

def init_model(args):
    i = 1 if args.gray else 3
    if args.model == 'C':
        model = C(c=args.crop)
    elif args.model == 'cMC':
        model = MC(
            block_size=args.filter_size, search_distance=8, branch=args.branch,
            c=args.crop
        )
    elif args.model == 'L':
        model = L(f=args.filter_size, t=2, i=i, c=args.crop)
    elif args.model == 'LNL':
        model = LNL(
            f=args.filter_size, t=2, i=i, k=args.num_channels, d=args.stride,
            c=args.crop, activation=args.nonlin, lmbda=args.lmbda,
            regularization=args.reg,
        )
    elif args.model == 'F':
        model = F(c=args.crop, branch=args.branch, epsilon=args.epsilon)
    elif args.model == 'Spyr':
        model = Spyr(
            image_shape=(args.image_size, args.image_size),
            num_ori=args.num_channels, num_scales=args.num_scales, c=args.crop,
            branch=args.branch, epsilon=args.epsilon,
        )
    elif args.model == 'PP':
        model = PP(
            f=args.filter_size, k=args.num_channels, i=i, d=args.stride,
            c=args.crop, mode=args.pad_mode, branch=args.branch,
            epsilon=args.epsilon, activation=args.nonlin, tied=args.tied 
        )
    elif args.model == 'mPP':
        model = mPP(
            f=args.filter_size, k=args.num_channels, i=i, d=args.stride,
            c=args.crop, mode=args.pad_mode, branch=args.branch,
            epsilon=args.epsilon, activation=args.nonlin, tied=args.tied,
            J=args.num_scales
        )
    elif args.model == 'QP':
        model = QP(
            f=args.filter_size, k=args.num_channels, i=i, d=args.stride,
            c=args.crop, mode=args.pad_mode, epsilon=args.epsilon,
            activation=args.nonlin, tied=args.tied, 
            group_size=args.group_size, num_quadratics=args.num_quadratics,
        )
    elif args.model == 'mQP':
        model = mQP(
            f=args.filter_size, k=args.num_channels, i=i, d=args.stride,
            c=args.crop, mode=args.pad_mode, epsilon=args.epsilon,
            activation=args.nonlin, tied=args.tied,
            group_size=args.group_size, num_quadratics=args.num_quadratics,
            J=args.num_scales
        )
    elif args.model == 'CNN':
        model = CNN(
            f=args.filter_size, k=args.num_channels, i=args.gray, j=args.depth,
            c=args.crop, branch=args.branch, activation=args.nonlin,
        )
    elif args.model == 'Unet':
        model = Unet(
            k=args.num_channels, i=args.gray, J=args.num_scales, c=args.crop,
        )
    else:
        print("incorrect model argument")
        model = None

    return model

def standardize(x, data_stats):
    """
    x [B C T H W] in range [0-1]
    data_stats {mean: [C], std [C]}
    """
    # reshaping for broadcasting
    C = x.shape[1]
    mean = data_stats['mean'].view(1, C, 1, 1, 1).to(x.device)
    std = data_stats['std'].view(1, C, 1, 1, 1).to(x.device)
    return (x - mean) / std

def undo_standardize(x, data_stats):
    """
    x [B C T H W]
    data_stats {mean: [C], std [C]}

    returns
    x in in range [0-1]
    """
    # reshaping for broadcasting
    C = x.shape[1]
    mean = data_stats['mean'].view(1, C, 1, 1, 1).to(x.device)
    std = data_stats['std'].view(1, C, 1, 1, 1).to(x.device)
    return x * std + mean

def SNR(target, pred):
    """
    relative to target
    [B, C, T, H, W] -> [B, T]
    """
    snr_vals = 10 * torch.log10(target.var(dim=(1, 3, 4)) /
                                (target - pred).var(dim=(1, 3, 4)))
    return snr_vals

def PSNR(target, pred, data_stats):
    """
    pred, target: floats of shape [B C T H W]
    data_stats {mean: [C], std [C]}

    returns
    PSNR values in dB, ie.log base 10
        shape [B T], ie. averaged over Channel, Height and Width.

    note:
    first undo standardization, so that signal range is 0-1.
    then compute:
        psnr =  10 log_10 (max^2 / mse)
             = -10 log_10 (mse)              since max is 1.
    """
    pred   = undo_standardize(pred, data_stats)
    target = undo_standardize(target, data_stats)
    psnr_vals = -10 * torch.log10((pred - target).pow(2).mean(dim=(1, 3, 4)))
    return psnr_vals

def SSIM(target, pred, data_stats, multiscale=False):
    """
    pred, target: floats of shape [B C T H W] in range [0-1]
    data_stats {mean: [C], std [C]}
    multiscale: option to compute ms_ssim (over 5 scales)
    
    returns
    SSIM values in [-1-1]
        shape [B T], ie. averaged over Channel, Height and Width.
    """
    pred = undo_standardize(pred, data_stats)
    target = undo_standardize(target, data_stats)
    pred_ = rearrange(pred, 'B C T H W -> (B T) C H W')
    target_ = rearrange(target, 'B C T H W -> (B T) C H W')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if multiscale:
            ssim_vals = po.metric.ms_ssim(pred_, target_)
        else:
            ssim_vals = po.metric.ssim(pred_, target_)
        ssim_vals = ssim_vals.mean(1)  # averaging ssim over channels
    ssim_vals = rearrange(ssim_vals, '(B T) -> B T', B=len(pred))
    return ssim_vals

def sort_by_ave_norm(W, g=2, plot_norm=False):
    """ Order groups of weights by their average norm.

    W: [C c H W] a cpu tensor w/o gradients
        C = (k g) k groups of size g
        c = 1 or 3 for RGB
    ----
    returns: W_sorted, the reordered weights
    """
    W_ = rearrange(W, '(k g) c H W -> k g c H W', g=g)
    W_norm = W_.pow(2).sum((2, 3, 4)).pow(.5)  # along color channel and space
    W_norm_ave = W_norm.mean(1)  # group mean
    idx = torch.argsort(W_norm_ave, descending=True)

    W_sorted = rearrange(W_[idx], 'k g c H W -> (k g) c H W')

    # W_norm = torch.linalg.norm(W.view(len(W), -1), axis=1)
    # W_norm_ave = (W_norm[::2] + W_norm[1::2]) / 2
    # id = np.argsort(W_norm_ave)
    # idx = np.zeros(len(W))
    # idx[::2] = id * 2
    # idx[1::2] = id * 2 + 1
    # idx = np.array(idx[::-1], dtype=int)

    if plot_norm:
        plt.figure()
        plt.plot(rearrange(W_norm[idx], 'k g -> (k g)', g=g), "o-")
        plt.title('weight norm')
        plt.show()

    return W_sorted
