import os
import glob

import numpy as np
import pandas as pd
from PIL import Image
from plenoptic.tools import blur_downsample
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from einops import rearrange

home = os.path.expanduser("~")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DAVIS(Dataset):
    def __init__(
            self, data_path=home + '/Documents/datasets/DAVIS', split="train",
            normalize=True, gray=True, image_size=32, clip_length=11,
            n_levels=1, n_crops=1, fold=2017, subset=0, overfit=0,
        ):
        """ Preprocess and load DAVIS video dataset.

        normalize: boolean
            mean subtract and divide by std (precomputed values)
        gray: boolean
            remove the color channel
        image_size: integer
            number of pixels in each spatial dimension (precomputed offline)
        clip_length: integer
            number of frames per video clip (precomputed offline)
        n_levels: 1, 2, 3
            number of downsamplings by a factor 2 (precomputed offline)
            increasing `n_levels` adds low frequency samples to the dataset
        n_crops: 1, 5, 9, 15, 21
            number of crops out of large video frames (precomputed offline)
            center, and 4 corners, and 4 cardinals, and 6 sides, and 6 edges
        fold: integer in [2017, range(9)]
            selection of train/test split:
            * 2017: predefined DAVIS/ImageSets/2017/train.txt
            * [0-8]: nine possible non-overlapping 80/10 splits of the data
        subset: int, optional, default 0
            if subset > 0, select `subset` of train clips
            ordered by how amount of motion, from median out
        overfit: bool, optional
            only cache the first train video for overfitting

        NOTES
        -----
        initialization
            . load dataset, or -if first time- preprocess and store dataset
            . cache daset to memory
            . apply transforms
        streaming (online, intended for pytorch dataloader with pinned memory)
            . load a clip
        preprocessing (offline, run once and store to disk)
            . remove color if `gray`
            . extract spatial crops of `image_size` pixels
            . downsample spatially `n_levels` times
            . chunk into clips of lenght `clip_length`
            . store data statistics for normalization

        TODO
        online augmentations with torchvision.transforms.RandomResizedCrop
        """
        super().__init__()

        self.raw_data_path = data_path
        self.split = split
        self.normalize = normalize
        self.gray = gray
        self.image_size = image_size
        assert n_levels <= (8 - np.log2(image_size)), print(
            f"""max n_levels given image_size is {int(8 - np.log2(image_size))},
                can not support {n_levels} levels."""
            )
        self.clip_length = clip_length
        assert n_crops in [1, 5, 9, 15, 21]
        self.n_crops = n_crops
        assert n_levels in [1, 2, 3]
        self.n_levels = n_levels
        self.fold = fold
        self.subset = subset
        self.overfit = overfit

        name = f"{image_size:03d}pixels"
        name += f"_{n_levels}levels"
        name += f"_{n_crops:02d}crops"
        name += "_gray" if gray else ""

        self.preprocessed_data_path = os.path.join(
            self.raw_data_path, "npySnips", name
        )
        print("data path ", self.preprocessed_data_path)
        if not os.path.exists(self.preprocessed_data_path):
            print(f"\n preprocessing dataset into {name}""")
            self.preprocess()

        self.select_folders()
        self.set_transforms()
        self.cache_data()

    def __len__(self):
        self.len = len(self.X)
        return self.len

    def __getitem__(self, index):
        x = self.X[index]
        return x

    def select_folders(self):
        """select list video folders for train (or val) split
        given a dataset fold (i.e. a partition into train/val)
        """
        if self.fold == 2017:
            # using predefined train/test split: 60 train videos, 30 test videos
            if self.split == "train":
                folders = pd.read_csv(os.path.join(
                    self.raw_data_path, "ImageSets", "2017", "train.txt"
                ), header=None)
            elif self.split == "val":
                folders = pd.read_csv(os.path.join(
                    self.raw_data_path, "ImageSets", "2017", "val.txt"
                ), header=None)
            else:
                print(f"split {self.split}, should be either train or val")
            self.folders = [f[0] for f in folders.values]

        elif self.fold in range(9):
            #  automated train/test split: 80 train videos, 10 test videos
            folders = sorted(glob.glob(os.path.join(
                self.raw_data_path, "JPEGImages", "480p", '*'
            )))
            folders = [os.path.basename(os.path.normpath(f)) for f in folders]
            if self.split == "train":
                l = [folders[i::9] for i in range(9) if i != self.fold]
                self.folders = [item for sublist in l for item in sublist]
            elif self.split == "val":
                self.folders = folders[self.fold::9]
            else:
                print(f"split {self.split}, should be either train or val")

    def set_transforms(self):
        """ subtract mean and divide by std
        else data in range [0, 1]
        """
        if self.normalize:
            try:
                # retrieve values from preprocessing
                data_stats = np.load(os.path.join(
                    self.preprocessed_data_path, "data_stats.npy"
                ), allow_pickle=True).item()
                self.mean = torch.from_numpy(data_stats['mean'])
                self.std = torch.from_numpy(data_stats['std'])
                self.transform = transforms.Compose([torch.from_numpy,
                                                     self.Normalize,
                                                    ])
            except FileNotFoundError:
                print("""`set_transforms` failed because the dataset
                      has not yet been preprocessed. Run the `preprocess()`
                      method first, and then rerun `set_transforms()`.""")
        else:
            # TODO naming is confusing
            self.mean = torch.zeros(1)
            self.std = torch.ones(1)
            self.transform = transforms.Compose([torch.from_numpy, ])

    def cache_data(self):
        """ cache data to memory at initialisation and apply transforms

        self.X: list of [C T H W] clips
    
        could be sped-up
        """
        try:
            X = []
            print(f"caching {self.split} clips data")
            # for each snip in given split (train/val)
            for folder in tqdm(self.folders):
                clips = sorted(glob.glob(os.path.join(
                    self.preprocessed_data_path, folder + "*"
                )))
                for s in clips:
                    snip = np.load(s)
                    snip = self.transform(snip)
                    X.append(snip)
                if self.overfit and self.split == 'train':
                    break
            if len(X) == 0:
                print("""`cache_data` failed because the dataset has not yet
                      been preprocessed. Run the `preprocess()` method first,
                      and then rerun `cache_data`. """)
            if self.split == 'train':
                self.X = self.select_subset(X)
            else:
                self.X = X
        except AttributeError:
            print("""`cache_data` failed because the transfroms have not yet
                  been set. Run the `set_transforms()` method first,
                  and then rerun `cache_data`.""")

    def Normalize(self, x):
        """ standardize sample x with data statistics

        manual implementation to fit tensor shape [C T H W]
        torchvision.transforms.Normalize assumes [C H W]
        """
        mean = self.mean.view(-1, 1, 1, 1)
        std = self.std.view(-1, 1, 1, 1)
        return x.sub_(mean).div_(std)

    def UnNormalize(self, x):
        """ undo the effect of `Normalize`
        """
        mean = self.mean.view(-1, 1, 1, 1).to(x.device)
        std = self.std.view(-1, 1, 1, 1).to(x.device)
        return x.mul_(std).add_(mean)

    def select_subset(self, X):
        """ pick clips with typical amount of motion

        option?
        make subset a fraction
        """
        subset = self.subset
        if subset != 0 and subset <= len(X):
            motion_energy = []
            for x in X:
                # rmse of diff as quick and dirty estimation of motion energy 
                me = torch.diff(x, dim=1).pow(2).mean((0,2,3)).pow(.5).mean()
                motion_energy.append(me)
            motion_energy = torch.tensor(motion_energy)
            idx = torch.argsort(motion_energy)
            ordered_X = torch.stack(X, 0)[idx]
            # mid = len(idx) // 2
            # if subset > 0:  # most motion
            #     self.X = [x for x in ordered_X[-subset:]]
            # if subset > 0:  # least motion
            #     self.X = [x for x in ordered_X[:subset]]
            if subset > 0: # from center out
                # subset = int(fraction * len(ordered_X))
                mid = len(ordered_X) // 2
                X = [x for x in
                            ordered_X[mid-subset//2: mid+subset//2+subset%2]]
            print(f"""Only using a subset of the data:
                      {len(X)} / {len(ordered_X)} clips.""")
        return X

    def preprocess(self):
        """preprocess data and store results to disk for future use

        create dir
        for each video
            load, convert, crop, rescale, downsample, chunk, save
        stats
        """
        os.makedirs(self.preprocessed_data_path)  #, exist_ok=True

        for split in ['val', 'train']:
            self.split = split
            self.select_folders()
            print(f"\n preprocessing {split} set""")

            videos = []
            num_dropped_frames = []
            # for each video in given split (train/val)
            for folder in tqdm(self.folders):
                video_paths = sorted(glob.glob(os.path.join(
                    self.raw_data_path, "JPEGImages", "480p", folder, "*.jpg"
                )))
                vid = self.load_vid(video_paths, device)
                for c in range(self.n_crops):
                    for n in range(self.n_levels):
                        name = folder + f"_c{c:02d}_n{n}_f"
                        n_dropped = self.save_clip(vid[n, c], name)
                videos.append(vid) # may be large
                num_dropped_frames.append(n_dropped)
            print(f"""dropped {sum(num_dropped_frames)} frames
                      due to clip length constraint.""")

            # save statistics (per channel)
            # list of length n_folders of [n_levels n_crops C T H W]
            # -> [n_levels n_crops C n_folders*T H W]
            Videos = np.concatenate(videos, axis=3)
            data_stats = {
                'mean': Videos.mean((0,1,3,4,5)),
                'std': Videos.std((0,1,3,4,5))
            }
            print('shape:', Videos.shape, '\n', data_stats,)
            if split == 'train':
                np.save(os.path.join(
                    self.preprocessed_data_path, "data_stats.npy"
                ), data_stats)
        # change flag back to inspect on small val set
        self.split = 'val'
        # load the corresponding folders
        self.select_folders()

    def load_vid(self, video_paths, device):
        """
        video_paths: list of paths to successive jpg images in a video

        vid: np.array of shape [n_levels n_crops C T H W]
            H=W=image_size
        """
        V = []
        for n in range(1, self.n_levels+1):
            S = int(self.image_size * 2 ** n)
            vid = np.empty(
                (self.n_crops, 1 if self.gray else 3, len(video_paths), S, S)
            )
            for t, im_path in enumerate(video_paths):
                # load
                im = self.load_im(im_path)
                # crop
                vid[:, :, t] = self.extract_crops(im, n_levels=n)
            # rescale dynamic range from [0, 255] to [0., 1.]
            vid = np.float32(vid / 255.)
            # downsample
            x = torch.from_numpy(vid).float().to(device)
            x = rearrange(x, 'B C T H W -> (B T) C H W')
            x = self.downsample(x, n_levels=n)
            x = rearrange(x, '(B T) C H W -> B C T H W', B=self.n_crops)
            V.append(np.float32(x.cpu().numpy()))
        return np.stack(V, axis=0)

    def load_im(self, im_path):
        """ load and convert to np.array

        im_path: path to jpg image on disk

        im: np.array of shape [C H W]
        """
        if self.gray:
            im = np.array(Image.open(im_path).convert('L'))
            im = np.expand_dims(im, axis=0) # [C H W]
        else:
            im = np.array(Image.open(im_path)) # [H W C]
            im = im.transpose((2,0,1)) # [C H W]
        return im

    def extract_crops(self, im, n_levels):
        """ extract overlapping patches

        im: np.array of shape [C H W] - large im
        n_levels: number of downsamplings
            extracts smaller patches for larger n

        im_crops: np.array of shape [B C h w] - small ims
            where B=n_crops, h=w=64*2**(n-1)

        For example:
        image_size=32
        n_levels=1, patch size  64x 64 downsampled 1 time  to 32x32
                 2, patch size 128x128 downsampled 2 times to 32x32
                 3, patch size 256x256 downsampled 3 times to 32x32
        n_crops= 1, focus on the center where motion happens
                 5, add four corners crops (partially overlapping)
                 9, four cardinal crops (fully overlapping = redundant)
                 15, six side crops
                 21, six edge crops
        """
        crops = [(0, 0)]
        if self.n_crops >= 5:
            crops.extend([(-1,-1), (-1, 1), (1, -1), (1, 1)])
        if self.n_crops >= 9:
            crops.extend([(-1, 0), (0, -1), (0, 1), (1, 0)])
        if self.n_crops >= 15:
            crops.extend([(-1,-2), (0,-2), (1,-2), (-1,2), (0,2), (-1,2)])
        if self.n_crops >= 21:
            crops.extend([(-1,-3), (0,-3), (1,-3), (-1,3), (0,3), (-1,3)])
        _, H, W = im.shape
        # print(H, W)
        S = int(self.image_size / 2 * 2 ** n_levels)
        s = 96  # controls overlap
        im_list = []
        for (i, j) in crops:
            hl = H//2-S+i*s
            hh = H//2+S+i*s
            wl = W//2-S+j*s
            wr = W//2+S+j*s
            assert hl >= 0, print(hl)
            assert hh <= H, print(hh, H)
            assert wl >= 0, print(wl)
            assert wr <= W, print(wr, W)
            # print(hl, hh, wl, wr)
            im_list.append(im[:, hl:hh, wl:wr])
        return np.stack(im_list, axis=0)

    def save_clip(self, vid, folder):
        """
        vid: [C T H W]

        save the clips [C t H W], where t is the clip length
        returns: number of dropped frames
        """
        # chunk into clips
        clip_overlap = 2  # the first two frames don't get predicted otherwise
        boundaries = np.arange(
            vid.shape[1]
        )[::(self.clip_length - clip_overlap)]
        for (i, b) in enumerate(boundaries):
            snip = vid[:, b:b+self.clip_length]
            # save each snip
            if snip.shape[1] == self.clip_length:
                np.save(os.path.join(
                    self.preprocessed_data_path, folder + f"{i:02d}.npy"
                ), snip)
        if snip.shape[1] == self.clip_length:
            return 0
        else:
            return snip.shape[1]

    def downsample(self, vid, n_levels=1):
        """
        vid: torch.tensor of shape [B C H W]

        vid: torch.tensor of shape [B C h w]
            h = H//2**-n_levels
            w = W//2**-n_levels
        """
        vid = blur_downsample(vid) / 2.
        if n_levels == 1:
            return vid
        else:
            return self.downsample(vid, n_levels=n_levels-1)
