import torch
import torch.nn as nn
from einops import rearrange
import torch.fft as fft


class C(nn.Module):
    """Copy the last observed frame (i.e. no-op prediction)

    c: int
        size of spatial crop (for comparison with other models)
    """
    def __init__(self, c=17):
        super().__init__()
        self.space_crop = c

    def predict(self, x):
        target = x[:, :, 2: ]
        pred   = x[:, :, 1:-1]
        x, x_hat = self.crop(target, pred, tau=0)
        return x, x_hat

    def crop(self, x, x_hat, tau=2):
        """ remove boundaries and initial frames
        prediction only valid for center picture and after a warmup period
        """
        H, W = x.shape[-2:]
        c = self.space_crop
        target = x[:, :, tau:, c:H-c, c:W-c]
        pred = x_hat[:, :, tau:, c:H-c, c:W-c]
        assert target.shape == pred.shape, f"{target.shape}, {pred.shape}"
        assert target.shape[-1] > 0, f"target shape {target.shape}"
        return target, pred


class MC(C):
    """MotionCompensation

    block-matching motion compensation with diamond search algorithm:
        Zhu, Shan and Ma, Kai-Kuang, A new diamond search algorithm for fast
        block-matching motion estimation, IEEE transactions on Image
        Processing 2000

    search_distance: int
        f
    block_size:
        f
    branch: str in ['causal', 'noncausal']
        f

    TODO
    - handle RGB
    - paralellize for speed-up
    """
    def __init__(self, search_distance=8, block_size=8,
                 branch='causal', c=17):
        super().__init__()
        from ppm.mocomp import blockMotion, flip, blockComp
        from numpy import stack
        self.space_crop = c
        self.search_distance = search_distance
        self.block_size = block_size
        self.branch = branch
        self.blockMotion = blockMotion
        self.flip = flip
        self.blockComp = blockComp
        self.stack = stack

    def predict(self, x):
        x = x.cpu()
        search_distance = self.search_distance
        block_size = self.block_size
        c = self.space_crop
        tau = 1

        x = rearrange(x, 'B C T H W -> B T H W C').numpy()
        # one clip at a time
        target, pred = [], []
        for videodata in x:

            motion = self.blockMotion(
                videodata, method="DS", mbSize=block_size, p=search_distance
            )
            motion = self.flip(motion)

            if self.branch == 'noncausal':
                compmotion = self.blockComp(
                    videodata, motion, mbSize=block_size
                )
                target.append(videodata[1:, c:-c, c:-c])
                pred.append(compmotion[1:, c:-c, c:-c])
            elif self.branch == 'causal':
                causal_compmotion = self.blockComp(
                    videodata[tau:], motion[:-tau], mbSize=block_size
                )
                target.append(videodata[tau:][1:, c:-c, c:-c])
                pred.append(causal_compmotion[1:, c:-c, c:-c])

        target, pred = self.stack(target, axis=0), self.stack(pred, axis=0)
        x = rearrange(
            torch.from_numpy(target), 'B T H W C -> B C T H W'
        ).float()
        x_hat = rearrange(
            torch.from_numpy(pred), 'B T H W C -> B C T H W'
        ).float()

        return x, x_hat


class Spyr(C):
    """Complex Steerable Pyramid

    Predict by extrapolating local phases in a fixed pyramid representation

    image_shape : `list or tuple` of two int
        shape of input image
    num_ori: int
        number of orientations in the pyramid
    num_scales: int, or 'auto'
        number of scales of the pyramid
    branch: str in ['phase', 'linear']
        choice of prediction mechanism
    activation: str in ['linear', 'amplitude', 'amplitude_linear']
        choice of nonlinearity
    downsample_fwd: boolean
        option to downsample spatially by a factor two
    """
    def __init__(
            self, image_shape, num_ori=4, num_scales='auto', c=17,
            branch='phase', epsilon=1e-10, activation='amplitude',
            downsample_fwd=False
        ):
        super().__init__(c=c)
        if image_shape is not None:
            from plenoptic.simulate import SteerablePyramidFreq
            num_scales = 'auto' if num_scales == 0 else num_scales
            self.pyr = SteerablePyramidFreq(
                image_shape, order=num_ori-1, height=num_scales, is_complex=True, 
                downsample=False, tight_frame=True
            )
            self.num_ori = num_ori
            self.num_scales = self.pyr.num_scales
            self.to = self.to_pyr
        self.space_crop = c
        self.epsilon = torch.tensor(epsilon)
        self.branch = branch
        self.activation = activation
        self.downsample_fwd = downsample_fwd

    def forward(self, x):
        y = self.analysis(x)
        x = self.nonlin(y)
        return x

    def predict(self, x):
        y = self.analysis(x)
        y_hat = self.advance(y)
        x_hat = self.synthesis(y_hat)
        x, x_hat = self.crop(x, x_hat)
        return x, x_hat

    def advance(self, z):
        T = z.shape[2]
        z_hat = torch.zeros_like(z)
        for t in range(1, T-1):
            z_hat[:, :, t+1] = self.tick(z[:, :, t-1], z[:, :, t])
        return z_hat

    def tick(self, p, c):
        "p: previous, c: current, f: future"
        if self.branch == 'phase':
            f = c ** 2 * p.conj() / torch.maximum(
                torch.abs(c) * torch.abs(p), self.epsilon
            )
        elif self.branch == 'linear':
            f = 2 * c - p
        return f

    def analysis(self, x):
        # NOTE: the pyramid expects 4 dimensional input tensors
        B = len(x)
        x = rearrange(x, "B C T H W -> (B T) C H W")
        c = self.pyr(x)
        y, self.info = self.pyr.convert_pyr_to_tensor(c)
        y = rearrange(y, "(B T) C H W -> B C T H W", B=B)
        return y

    def synthesis(self, y):
        # NOTE: pyramid reconstruction uses real part only
        B = len(y)
        y = rearrange(y, "B C T H W -> (B T) C H W")
        c = self.pyr.convert_tensor_to_pyr(y.real, *self.info)
        x = self.pyr.recon_pyr(c)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def nonlin(self, y):
        if self.activation is None:
            x = y
        elif self.activation == 'linear':
            x = y.real
        elif self.activation == 'amplitude':
            x = torch.abs(y)
        elif self.activation == 'amplitude_linear':
            x = torch.cat((y.real, torch.abs(y)), dim=1)
        if self.downsample_fwd:
            x = x[..., ::2, ::2]
        return x

    def to_pyr(self, *args, **kwargs):
        self.pyr = self.pyr.to(*args, **kwargs)
        return self


### extras ###
class F(Spyr):
    """Fourier predictor
    
    Predict by extrapolating spatial phases in the frequency representation
    """
    def __init__(self, c=17, branch='phase', epsilon=1e-10):
        super().__init__(image_shape=None, c=c, branch=branch, epsilon=epsilon)
        self.space_crop = c
        self.branch = branch
        self.epsilon = torch.tensor(epsilon)

    def forward(self, x):
        y = self.analysis(x)
        x = torch.abs(y)
        return x

    def analysis(self, x):
        z = fft.fft2(x, dim=(-2,-1), norm='ortho')
        return z

    def synthesis(self, z):
        x = fft.ifft2(z, dim=(-2,-1), norm='ortho').real
        return x


class LE(C):
    """Linear Extrapolation
    
    Predict via straight extrapolation of the signal space
    """
    def __init__(self, c=17):
        super().__init__(c=c)

    def forward(self, x):
        return self.predict(x)

    def predict(self, x):
        target = x[:, :, 2:]
        current = x[:, :, 1:-1]
        previous = x[:, :, :-2]
        pred = 2*current - previous
        x, x_hat = self.crop(target, pred, tau=0)
        return x, x_hat


class L(C):
    """Linear predictor

    f: int, odd
        spatial size of filter
    t: int
        number of observed frames, memory length
    i: int
        number of channels in
    c: int
        size of spatial crop
    """
    def __init__(self, f=17, t=2, i=1, c=17,):
        super().__init__(c=c)
        assert f % 2 == 1
        p = (0, f//2, f//2)
        self.W = nn.Conv3d(i, i, (t, f, f), padding=p, bias=False)

    def predict(self, x):
        x_hat = self.W(x)
        x, x_hat = self.crop(x, x_hat)
        return x, x_hat

    def crop(self, x, x_hat):
        H, W = x.shape[-2:]
        c = self.space_crop
        t = self.W.weight.shape[-3]
        target = x[:, :, t:, c:H-c, c:W-c]
        pred = x_hat[:, :, :-1, c:H-c, c:W-c]
        assert target.shape == pred.shape, f"{target.shape}, {pred.shape}"
        assert target.shape[-1] > 0, f"target shape {target.shape}"
        return target, pred


class LNL(L):
    """Linear Non-linear Linear predictor

    k: int
        number of channels
    d: int
        downsampling factor
    activation: str in ['relu', 'softmax']
        choice of non-linearity
    """
    def __init__(
            self, f=17, t=2, i=1, k=50, d=1, c=17, activation='relu',
        ):
        super().__init__()
        self.space_crop = c

        self.W = nn.Conv3d(i, k, (t, f, f), bias=True, stride=(1, d, d))
        self.w = nn.ConvTranspose3d(
            k, i, (1, f, f), bias=False, stride=(1, d, d)
        )
        # initialize with small values (rich regime)
        self.W.weight.data /= 100
        self.W.bias.data *= 0
        self.w.weight.data /= 100
        if activation == 'relu':
            self.nonlin = nn.ReLU()
        elif activation == 'softmax':
            self.nonlin = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.W(x)
        return self.nonlin(y)

    def predict(self, x):
        y = self.forward(x)
        x_hat = self.w(y)
        x, x_hat = self.crop(x, x_hat)
        return x, x_hat
