import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


class CNN(nn.Module):
    """CNN prediction model

    The network maps two observed frames to a predicted upcoming frame.
    Prediction is computed in forward pass, there is no representation space.

    f: int
        spatial size of filter
    k: int
        number of channels
    i: int
        number of channels in
    c: int
        size of spatial crop
    j: int
        number of layers
    branch: str in ['residual', 'vanilla']
        choice of architecture x_t+1_hat = current + CNN(x_t, x_t-1)
                            or x_t+1_hat = CNN(x_t, x_t-1)

    TODO
    argument to toggle bias
    add argument for memory length tau
    """
    def __init__(
            self, f=3, k=64, i=1, c=17, j=16, branch='residual',
            activation='relu', batchnorm=True, learnable_sd=True,
        ):
        super().__init__()
        assert j >= 2
        self.space_crop = c
        self.branch = branch
        self.activation = activation
        self.net = Net(K_size=f, C_in=i, C_hid=k, C_out=i, N_layer=j,
                       K_0=(2, f, f), activation=activation,
                       batchnorm=batchnorm, learnable_sd=learnable_sd)

    def forward(self, y):
        if self.branch == 'vanilla':
            x_hat = self.net(y)
        elif self.branch == 'residual':
            x_hat = y[:, :, -1:] + self.net(y)
        return x_hat

    def predict(self, x):
        _, y = self.patchify(x)
        x_hat = self.forward(y)
        x_hat = self.unpatchify(x_hat, len(x))
        x_star = x[:, :, 2:] # == self.unpatchify(_, x.shape)
        target, pred = self.crop(x_star, x_hat)
        return target, pred

    def patchify(self, x):
        """Stack all groups of three successive frames along batch dimension,
        and split between 1 target frame and 2 observed frames.

        x: B C T H W
        --
        # taget_patches: B*(T-2) C 1 H W
        past_patches : B*(T-2) C 2 H W
        """
        T = x.shape[2]
        X = torch.stack([x[:, :, t:t+3] for t in range(T-2)])
        X = rearrange(X, 's b c t h w -> (s b) c t h w')
        return X[:, :, -1:], X[:, :, :2]

    def unpatchify(self, patches_slice, B):
        """Rearrange into clips (5d tensors)

        patches_slice: B*(T-2) C 1 H W
        --
        x: B C (T-2) H W
        """
        x = rearrange(patches_slice, '(s b) c 1 h w -> b c s h w', b=B)
        return x

    def crop(self, x, x_hat):
        c = self.space_crop
        H, W = x.shape[-2:]
        target = x[:, :, :, c:H-c, c:W-c]
        pred = x_hat[:, :, :, c:H-c, c:W-c]
        assert target.shape == pred.shape, f"{target.shape}, {pred.shape}"
        assert target.shape[-1] > 0, f"target shape {target.shape}"
        return target, pred


class Net(nn.Module):
    """CNN
    
    bias free, homogeneous order 1

    K_0: tuple of three ints, optional
        Specifies shape of kernel in first layer spatio-temporal convolution.
        The following layers compute spatial convolutions and treat time
        as a batch dimension.


    NOTE
    ----
    input:   B C_in  T H W
    N_layer: B C_hid t H W
        conv (bias free)
        batchnorm (optional)
        activation (relu, etc.)
        residual (optional)
    output:  B C_out t H W
    """
    def __init__(
            self, K_size, C_in, C_hid, C_out, N_layer, K_0=None, residual=False,
            activation='relu', batchnorm=True, learnable_sd=True,
        ):
        super().__init__()
        if K_0 is None:
            K_0 = K_size
        self.net = nn.Sequential(
            Conv(C_in, C_hid, K_0, activation=activation,
                 batchnorm=batchnorm, learnable_sd=learnable_sd),
            *[Conv(C_hid, C_hid, K_size, activation=activation,
                   batchnorm=batchnorm, learnable_sd=learnable_sd,
                   residual=residual)
            for j in range(1, N_layer - 1)],
            Conv(C_hid, C_out, K_size, activation=False,
                 batchnorm=False, learnable_sd=False)
        )

    def forward(self, x):
        for j in range(len(self.net)):
            x = self.net[j](x)
        return x


class Conv(nn.Module):
    """ Conv->BN->nonlin->residual

    kernel_size: int or tuple
        tuple: 3d conv with kernel (int, int, int)
        or int: 2d conv with kernel (1, int, int), dummy time dim

    """
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, mode='same',
        transposed=False, activation='relu', residual=False, batchnorm=True,
        learnable_sd=True,
        ):
        super().__init__()

        if isinstance(kernel_size, int):
            assert kernel_size % 2 == 1
            if mode == 'same':
                padding = (0, kernel_size // 2, kernel_size // 2)
            elif mode == 'valid':
                padding = (0, 0, 0)
            kernel_size = (1, kernel_size, kernel_size)
            out_pad = (0, stride // 2, stride // 2)
            stride = (1, stride, stride)
        elif isinstance(kernel_size, tuple) and len(kernel_size) == 3:
            # still no temporal padding or stride
            assert kernel_size[-1] % 2 == 1
            if mode == 'same':
                padding = (0, kernel_size[1] // 2, kernel_size[2] // 2)
            elif mode == 'valid':
                padding = (0, 0, 0)
            out_pad = (0, stride // 2, stride // 2)
            stride = (1, stride, stride)

        if not transposed:
            self.conv = nn.Conv3d(in_channels, out_channels,
               kernel_size=kernel_size, bias=False, stride=stride,
               padding=padding)
        else:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                kernel_size=kernel_size, bias=False, stride=stride,
                padding=padding, output_padding=out_pad)
        modules = [('conv', self.conv)]

        self.batchnorm = batchnorm
        if batchnorm:
            self.norm = BF_BN(
                out_channels=out_channels, learnable_sd=learnable_sd
            )
            modules.append(('batchnorm', self.norm))

        self.activation = activation
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation is False:
            self.act = nn.Identity()
        else:
            raise NotImplementedError
        modules.append(('activation', self.act))

        self.residual = residual
        if residual:
            assert in_channels == out_channels

        self._initialize_weights()

    def forward(self, x):
        y = self.conv(x)
        if self.batchnorm:
            y = self.norm(y)
        y = self.act(y)
        if self.residual:
            y = x + y
        return y

    def _initialize_weights(self):
        with torch.no_grad():
            if self.activation == 'relu':
                nn.init.kaiming_normal_(self.conv.weight, a=0, mode='fan_in')


class BF_BN(nn.Module):
    """
    BatchNorm3d without bias
    """
    def __init__(self, out_channels, learnable_sd=False):
        super().__init__()
        self.learnable_sd = learnable_sd
        self.running_sd = nn.Parameter(
            torch.ones(1, out_channels, 1, 1, 1), requires_grad=False)
        if learnable_sd:
            self.gamma = nn.Parameter(
                torch.ones(1, out_channels, 1, 1, 1), requires_grad=True)

    def forward(self, x):
        var_x = x.var(dim=(0, 2, 3, 4), keepdim=True, unbiased=False)
        epsilon = torch.tensor(1e-05)
        sd_x = torch.sqrt(torch.maximum(var_x, epsilon))

        if self.training:
            x = x / sd_x
            self.running_sd.data = (1 - 0.1) * self.running_sd.data + 0.1 * sd_x
        else:
            x = x / self.running_sd
        if self.learnable_sd:
            x = x * self.gamma
        return x


class Unet(CNN):
    """U-Net

    J: int
        number of (nonlinear) scales

    - model specific arguments
    - basepredictor superclass
    """
    def __init__(self, k=64, i=1, J=4, c=17,
                 branch='residual'):
        super().__init__()
        self.space_crop = c
        self.branch = branch
        self.net = UNet(
            in_channels=i, width=k, out_channels=i, J=J, K_0=(2, 3, 3)
        )


class UNet(nn.Module):
    """
    - narrow: constant width, 64 channels at all scales
    - temporal signal, spatial processing:
        Conv3d, BatchNorm3d, MaxPool3d, wrap Upsample
    - bias-free:
        including batch norm (leaking time info?), upsampling and out conv

    width: int
        number of channels at intermediate layers
    J: int
        number of scales
        eg. j=1 is no downsamplings: 2 convolutions on input and one on output
            j=5 is 4 downsamplings: 2 + 8 + 8 + 1 convolutions
    TODO
    analysis_last_ReLU: bool
        option to omit the last ReLU of the analysis transform,
        so that coefficients are unconstrained (for Polar Prediction of latents)
    """
    def __init__(
            self, in_channels, out_channels=None, width=64, J=4, K_0=(2, 3, 3),
            analysis_last_ReLU=True
        ):
        super().__init__()
        if not out_channels:
            out_channels = in_channels
        # originally [in_channel, 64, 128, 256, 512, 1024]
        self.n = [in_channels] + [width] * J
        self.inc = DoubleConv(self.n[0], self.n[1], K_0=K_0,
                              last_ReLU=analysis_last_ReLU)
        self.downs = nn.ModuleList([Down(self.n[i], self.n[i+1],
                                         last_ReLU=analysis_last_ReLU)
                                    for i in range(1, len(self.n) - 1)])
        self.ups = nn.ModuleList([Up(self.n[i+1] + self.n[i], self.n[i])
                                  for i in range(len(self.n) - 2, 0, -1)])
        self.outc = OutConv(self.n[1], out_channels)

    def analysis(self, x):
        y_list = [self.inc(x)]
        for f in self.downs:
            y_list.append(f(y_list[-1]))
        return y_list

    def synthesis(self, y_list):
        x = y_list.pop()
        for f in self.ups:
            x = f(x, y_list.pop())
        x = self.outc(x)
        return x

    def forward(self, x):
        y_list = self.analysis(x)
        x_hat = self.synthesis(y_list)
        return x_hat


class BFBatchNorm3d(nn.Module):
    """
    Note: this is a BatchNorm with no running mean and no learnable weight/bias
    """
    def __init__(self, num_features):
        super().__init__()
        self.register_buffer("running_sd", torch.ones(1, num_features, 1, 1, 1))
    def forward(self, x):
        if self.training:
            sd_x = torch.sqrt(
                x.var(dim=(0, 2, 3, 4), keepdim=True, correction=0) + 1e-05
            )
            x = x / sd_x
            self.running_sd.data = 0.9 * self.running_sd.data + 0.1 * sd_x
        else:
            x = x / self.running_sd
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None,
                K_0=None, last_ReLU=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.last_ReLU = last_ReLU
        if K_0 is None:
            K_0 = (1, 3, 3)
        self.conv1 = nn.Conv3d(
            in_channels, mid_channels, kernel_size=K_0, padding=(0, 1, 1),
            bias=False
        )
        self.batchnorm1 = BFBatchNorm3d(mid_channels)
        self.conv2 = nn.Conv3d(
            mid_channels, out_channels, kernel_size=(1, 3, 3),
            padding=(0, 1, 1), bias=False
        )
        self.batchnorm2 = BFBatchNorm3d(out_channels)
    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.batchnorm2(self.conv2(x))
        if self.last_ReLU:
            x = F.relu(x)
        # else omit the last ReLU
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, last_ReLU=True):
        super().__init__()
        self.maxpool = nn.MaxPool3d((1, 2, 2))
        self.conv = DoubleConv(in_channels, out_channels, last_ReLU=last_ReLU)
    def forward(self, x):
        return self.conv(self.maxpool(x))


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                Rearrange('b c t h w -> b (c t) h w'),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                Rearrange('b (c t) h w -> b c t h w', c=in_channels//2)
            )
        else:
            # option to learn the upsampling parameters
            # should be a single 2x2 sconvolution, but here separate per channel
            self.up = nn.ConvTranspose3d(
                in_channels // 2, in_channels // 2, groups=in_channels // 2,
                kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False
            )
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
    def forward(self, x1, x2):
        x1 = self.up(x1)  # [B C T H W]
        diffY = x2.size()[-2] - x1.size()[-2]
        diffX = x2.size()[-1] - x1.size()[-1]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # lean (less undesirable extra parameters)
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        # originally: 
        # self.conv = DoubleConv(
        #     in_channels, out_channels, in_channels, last_ReLU=False
        # )

    def forward(self, x):
        return self.conv(x)
