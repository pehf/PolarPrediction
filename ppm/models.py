import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from plenoptic.simulate import LaplacianPyramid

class PP(nn.Module):
    """ Polar Prediction Model

    Predict by extrapolating local phases of learned convolutional tight frame

    f: int
        spatial size of filter
    k: int
        number of pairs of channels
    i: int
        number of channels in
    d: int
        downsampling factor (stride of the convolution)
    c: int
        size of spatial crop
    mode: str in ['valid', 'same']
        spatial zero padding
    branch: str in ['phase', 'phase_cmplx']
        choice of real or complex valued implementation of polar prediction,
        which are identical up to machine precision
    epsilon: float
        stability of division
    activation: str in ['linear', 'amplitude', 'amplitude_linear']
        choice of nonlinearity
    tied: boolean
        sharing analysis and synthesis weights
    """
    def __init__(
            self, f=17, k=16, i=1, d=1, c=17, mode='valid', branch='phase',
            epsilon=1e-10, activation="amplitude", init_scale=1, tied=True,
        ):
        super().__init__()

        W = torch.randn(k*2, i, 1, f, f) / f**2
        self.W = nn.Parameter(W * init_scale)
        self.tied = tied
        if not tied:
           V = torch.randn(k*2, i, 1, f, f) / f**2
           self.V = nn.Parameter(V * init_scale)

        self.stride = (1, d, d)
        self.space_crop = c
        if mode == 'valid':
            self.padding = (0, 0, 0)
            self.out_padding = (0, 0, 0)
        elif mode == 'same':
            self.padding = (0, f//2, f//2)
            self.out_padding = (0, d-1, d-1)

        if branch == 'phase':
            self.tick = self.tick_real
        elif branch == 'phase_cmplx':
            self.tick = self.tick_cmplx

        self.epsilon = torch.tensor(epsilon)
        self.activation = activation

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

    def analysis(self, x):
        return F.conv3d(x, self.W, stride=self.stride, padding=self.padding)

    def synthesis(self, y):
        if self.tied:
            W = self.W
        else:
            W = self.V
        return F.conv_transpose3d(
            y, W, stride=self.stride, padding=self.padding,
            output_padding=self.out_padding
        )

    def advance(self, y):
        T = y.shape[2]
        y_hat = torch.zeros_like(y)
        for t in range(1, T-1):
            y_hat[:, :, t+1] = self.tick(y[:, :, t-1], y[:, :, t])
        return y_hat

    def tick_real(self, p, c):
        delta = self.mult(self.norm(c), self.conj(self.norm(p)))
        f = self.mult(delta, c)
        return f

    def mult(self, a, b):
        c = torch.empty_like(a, device=a.device)
        # plain
        # c[:,  ::2] = a[:,  ::2] * b[:, ::2] - a[:, 1::2] * b[:, 1::2]
        # c[:, 1::2] = a[:, 1::2] * b[:, ::2] + a[:,  ::2] * b[:, 1::2]
        # Gauss's trick
        one = a[:,  ::2] * b[:, ::2]
        two = a[:, 1::2] * b[:, 1::2]
        three = (a[:,  ::2] + a[:, 1::2]) * (b[:, ::2] + b[:, 1::2])
        c[:,  ::2] = one - two
        c[:, 1::2] = three - one - two
        return c

    def norm(self, x):
        x_ = torch.empty_like(x, device=x.device)
        n = (x[:, ::2] ** 2 + x[:, 1::2] ** 2 + self.epsilon) ** .5
        x_[:, ::2], x_[:, 1::2] = x[:, ::2] / n, x[:, 1::2] / n
        return x_

    def conj(self, x):
        x[:, 1::2] = -x[:, 1::2]
        return x

    def tick_cmplx(self, p, c):
        p = self.rect2pol(p)
        c = self.rect2pol(c)
        delta = c * p.conj() / torch.maximum(
            torch.abs(c) * torch.abs(p), self.epsilon
        )
        f = delta * c
        f = self.pol2rect(f)
        return f

    def rect2pol(self, y):
        return torch.complex(y[:, ::2], y[:, 1::2])

    def pol2rect(self, z):
        return rearrange(
            torch.stack((z.real, z.imag), dim=1), 'b c k h w-> b (k c) h w'
        )

    def nonlin(self, y):
        if self.activation is None:
            x = y
        elif self.activation == 'linear':
            x = y.real
        elif self.activation == 'amplitude':
            x = torch.abs(y)
        elif self.activation == 'amplitude_linear':
            x = torch.cat((y.real, torch.abs(y)), dim=1)
        return x

    def autoencode(self, x):
        return self.synthesis(self.analysis(x))

    def crop(self, x, x_hat, tau=2):
        """
        prediction only valid for center picture and after a warmup period
        """
        H, W = x.shape[-2:]
        c = self.space_crop
        target = x[:, :, tau:, c:H-c, c:W-c]
        pred = x_hat[:, :, tau:, c:H-c, c:W-c]
        assert target.shape == pred.shape, f"{target.shape} {pred.shape}"
        assert target.shape[-1] > 0, f"target shape {target.shape}"
        return target, pred


class mPP(PP):
    """multiscale Polar Prediction Model

    spatial filtering and temporal processing of fixed Laplacian pyramid
    coefficients, same learned filters applied at each scale
    
    J: int
        number of scales
    see documentation of PP for other arguments

    NOTE
    - explicit downsampling for speed
    """
    def __init__(
            self, f=17, k=16, i=1, d=1, c=17, mode='valid', branch='phase',
            epsilon=1e-10, activation="amplitude", init_scale=1, tied=True, J=4
        ):
        super().__init__(
            f=f, k=k, i=i, d=d, c=c, mode=mode, branch=branch, epsilon=epsilon,
            activation=activation, init_scale=init_scale, tied=tied
        )

        self.lpyr = LaplacianPyramid(J)

    def predict(self, x):
        y = self.analysis_pyr(x)
        y_hat = [self.advance(y) for y in y]
        x_hat = self.synthesis_pyr(y_hat)
        x, x_hat = self.crop(x, x_hat)
        return x, x_hat

    def analysis_pyr(self, x):
        # NOTE: the pyramid expects 4 dimensional input tensors
        B = len(x)
        x = rearrange(x, "B C T H W -> (B T) C H W")
        y = self.lpyr(x)
        y = [rearrange(y, "(B T) C H W -> B C T H W", B=B) for y in y]
        y = [self.analysis(y) for y in y]
        return y

    def synthesis_pyr(self, y):
        y = [self.synthesis(y) for y in y]
        B = len(y[0])
        y = [rearrange(y, "B C T H W -> (B T) C H W") for y in y]
        x = self.lpyr.recon_pyr(y)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def to(self, device):
        new_self = super(mPP, self).to(device)
        new_self.lpyr = new_self.lpyr.to(device)
        return new_self


class QP(PP):
    """Quadratic Prediction Model

    Prediction via learned quadratic prediction mechanism:
        - not imposing phase advance
        - generalizing beyond pairs

    group_size: int
        number of channels in each group, two is sufficient for diagonalizing
        commutative actions, and here it can be larger
    num_quadratics: int
        number of quadratic units in quadratic prediction mechanism, defaults
        to eight times group_size
    init_on_pp: boolean
        initialize weights of the quadratic prediction mechanism on
        the polar extrapolation solution for debugging purposes
    see documentation of PP for other arguments

    TODO
    canonicalize weights norm
    """
    def __init__(
            self, f=17, k=16, i=1, d=1, c=17, mode='valid',
            epsilon=1e-10, activation="amplitude", init_scale=1, tied=True,
            group_size=2, num_quadratics=None, init_on_pp=False
        ):
        super().__init__(
            f=f, k=k, i=i, d=d, c=c, mode=mode, epsilon=epsilon,
            activation=activation, init_scale=init_scale, tied=tied
        )

        self.group_size = group_size
        W = torch.randn(k*group_size, i, 1, f, f) / f**2
        self.W = nn.Parameter(W * init_scale)

        if num_quadratics is None:
            num_quadratics = group_size * 8
        self.L1 = nn.Linear(group_size*2, num_quadratics, bias=False)
        self.L2 = nn.Linear(num_quadratics, group_size*group_size, bias=False)
        if init_on_pp:
            self.init_on_phase_extrapolation(group_size, num_quadratics)

        self.tick = self.tick_learned_quadratic

    def tick_learned_quadratic(self, p, c):
        """Apply predictor matrix to current state

        [B C H W], [B C H W] -> [B C H W]
            where C = (K G) 
            i.e. num_channels = num_groups x group_size
        """
        # get the predictor matrix
        p_groups = rearrange(p, 'B (K G) H W -> B K H W G', G=self.group_size)
        c_groups = rearrange(c, 'B (K G) H W -> B K H W G', G=self.group_size)
        M = self.delta(p_groups, c_groups)

        # batch matrix multiply
        f_hat_ = torch.matmul(M, c_groups.unsqueeze(-1)).squeeze(-1)
        f_hat = rearrange(f_hat_, 'B K H W G -> B (K G) H W')
        return f_hat

    def delta(self, p_groups, c_groups):
        """Compute predictor matrix
        via normalization and Linear-Square-Linear (LSL) cascade

        [B K H W G], [B K H W G] -> [B K H W G G]
        """
        # unitarize
        p_amplit = p_groups.pow(2).sum(dim=-1, keepdim=True).pow(.5)
        c_amplit = c_groups.pow(2).sum(dim=-1, keepdim=True).pow(.5)
        p_unit = p_groups / torch.maximum(p_amplit, self.epsilon)
        c_unit = c_groups / torch.maximum(c_amplit, self.epsilon)
        y_unit = torch.concatenate((c_unit, p_unit), dim=-1) # B K H W (T G)
        # LSL: linear encoding, point-wise non-linearity, linear decoding
        m = self.L2(self.L1(y_unit) ** 2)
        # matricize
        M = rearrange(m, 'B K H W (G g) -> B K H W G g', G=self.group_size)
        return M

    def init_on_phase_extrapolation(self, group_size, num_quadratics):
        assert group_size == 2
        assert num_quadratics == 8
        w1 = torch.tensor(
            ((1, 0, 1, 0),
             (1, 0, -1, 0),
             (0, 1, 0, 1),
             (0, 1, 0, -1),
             (0, 1, 1, 0),
             (0, -1, 1, 0),
             (1, 0, 0, 1),
             (1, 0, 0, -1))
        ) * (1/4)**(1/3)
        w2 = torch.tensor(
            ((1, -1, 1, -1, 0, 0, 0, 0),
             (0, 0, 0, 0, -1, 1, 1, -1),
             (0, 0, 0, 0, 1, -1, -1, 1),
             (1, -1, 1, -1, 0, 0, 0, 0),)
        ) * (1/4)**(1/3)
        self.L1.weight.data = w1
        self.L2.weight.data = w2

    # def canonicalize(self):
    #     """
    #     TODO - fix
    #     enforce unit gain of the predictor,
    #     and rescale the analysis/synthesis weights accordingly.
    #     """
    #     with torch.no_grad():
    #        y = torch.randn(
    #           2**10, self.group_size, 2, 1, 1
    #        ).float().to(self.W.device)
    #         ratio = (
    #             y[:,:,1].pow(2).sum(1).pow(.5) /
    #             self.tick_learned_quadratic(
    #                 y[:, :, 0], y[:, :, 1]
    #             ).pow(2).sum(1).pow(.5)
    #         )   
    #         gamma = ratio.mean()
    #         self.L1.weight.data *= gamma ** (1/3)
    #         self.L2.weight.data *= gamma ** (1/3)
    #         self.W.data /= gamma ** (1/2)
    #     return self


class mQP(QP):
    """multiscale Quadratic Prediction Model

    J: int
        number of scales
    see documentation of QP for other arguments

    """
    def __init__(
            self, f=17, k=16, i=1, d=1, c=17, mode='valid',
            epsilon=1e-10, activation="amplitude", init_scale=1, tied=True,
            group_size=2, num_quadratics=6, J=4
        ):
        super().__init__(
            f=f, k=k, i=i, d=d, c=c, mode=mode, epsilon=epsilon,
            activation=activation, init_scale=init_scale, tied=tied,
            group_size=group_size, num_quadratics=num_quadratics
        )

        self.lpyr = LaplacianPyramid(J)

    def predict(self, x):
        y = self.analysis_pyr(x)
        y_hat = [self.advance(y) for y in y]
        x_hat = self.synthesis_pyr(y_hat)
        x, x_hat = self.crop(x, x_hat)
        return x, x_hat

    def analysis_pyr(self, x):
        # NOTE: the pyramid expects 4 dimensional input tensors
        B = len(x)
        x = rearrange(x, "B C T H W -> (B T) C H W")
        y = self.lpyr(x)
        y = [rearrange(y, "(B T) C H W -> B C T H W", B=B) for y in y]
        y = [self.analysis(y) for y in y]
        return y

    def synthesis_pyr(self, y):
        y = [self.synthesis(y) for y in y]
        B = len(y[0])
        y = [rearrange(y, "B C T H W -> (B T) C H W") for y in y]
        x = self.lpyr.recon_pyr(y)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def to(self, device):
        new_self = super(mQP, self).to(device)
        new_self.lpyr = new_self.lpyr.to(device)
        return new_self
