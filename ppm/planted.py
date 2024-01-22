import numpy as np
import torch
from torchvision.transforms.functional import (
    center_crop, rotate, resized_crop, InterpolationMode
)
import plenoptic as po


def sample_translations(x, f, T, vmax=5, v=None, strict=True):
    """if strict: circular convolution
       else: open boundary condition, ie. new content enters/exits at boundary
    """
    b, C, tau, H, W = x.shape
    X = x.view(b * tau, C, 1, H, W)
    B = len(X)
    Y = torch.zeros(B, C, T, f, f)

    if strict:
        for b in range(B):
            x_patch = center_crop(X[b], output_size=(f, f))
            if v is not None:
                vx, vy = np.random.choice(v, 2)
            else:
                vx, vy = np.random.randint(-vmax, vmax + 1, (2,))
            for t in range(T):
                Y[b, :, t] = torch.roll(
                    x_patch, shifts=(t * vx, t * vy), dims=(-2, -1)
                )
    else:
        for b in range(B):
            if v is not None:
                vx, vy = np.random.choice(v, 2)
            else:
                vx, vy = np.random.randint(-vmax, vmax + 1, (2,))
            min = np.maximum(0, -((T-1) * vx))
            max = np.minimum(H, H - ((T-1) * vx))
            x0 = np.random.randint(min, max - f + 1)
            
            min = np.maximum(0, -((T-1) * vy))
            max = np.minimum(W, W - ((T-1) * vy))
            y0 = np.random.randint(min, max - f + 1)

            for t in range(T):
                x = x0 + t * vx
                y = y0 + t * vy
                Y[b, :, t] = X[b, :, 0, x : x + f, y : y + f]
    return Y

def sample_rotations(x, f, T, amax=45, a=None, strict=True):
    """if strict: mask out corners
       else: new content creeps in at corners
    """
    b, C, tau, H, W = x.shape
    X = x.view(b * tau, C, 1, H, W)
    B = len(X)
    X = center_crop(X, (int(2 ** 0.5 * f), int(2 ** 0.5 * f)))
    Y = torch.zeros(B, C, T, f, f)
    for b in range(B):
        if a is not None:
            theta = np.random.choice(a)
        else:
            theta = np.random.randint(-amax, amax + 1)

        for t in range(T):
            Y[b, :, t] = center_crop(rotate(
                X[b], angle=theta * t, interpolation=InterpolationMode.BILINEAR
            ), output_size=(f, f))
    if strict:
        disk = po.tools.make_disk(f, f/2, f/2)
        Y = Y * disk.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return Y

def sample_zooms(x, f, T, zmax=3):
    """Note: bilinear interpolation is quite poor
    TODO: current implementation is going too fine scale
    """
    b, C, tau, H, W = x.shape
    X = x.view(b * tau, C, 1, H, W)
    B = len(X)
    Y = torch.zeros(B, C, T, f, f)
    s = (96 - f) // 2
    for b in range(B):
        z = np.random.randint(1, zmax+1)
        seq = torch.cat([resized_crop(
            X[b], top=s-t, left=s-t, height=f+2*t, width=f+2*t, size=(f, f),
            interpolation=InterpolationMode.BILINEAR
        ) for t in np.arange(s, s-T*z, -z)], dim=1)
        # equal proba zoom-in or zoom-out
        if np.random.rand() > .5:
            seq = torch.flip(seq, (1,))
        Y[b] = seq

    return Y
