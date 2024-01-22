import numpy as np
import matplotlib.pyplot as plt
import torch
import plenoptic as po
from ppm.utils import sort_by_ave_norm


def visualize_prediction_dynamics(model, x, device):
    """
    x: [B, C, T H, W]
    
    TODO this needs some work!
    pick some informative example (not merely first minibatch)
    choice of j and t, multiple images
        x = dataset[j].unsqueeze(0).to(device)
    """
    j = 0
    t = 1
    x = x[j:j+1].to(device)
    target, pred = model.predict(x)
    curr = target.cpu()[:, :, t-1]
    target = target.cpu()[:, :, t]
    pred = pred.detach().cpu()[:, :, t]
    # PSNR would be more informative
    mse = (pred - target).pow(2).mean()
    copy_mse = (curr - target).pow(2).mean()
    as_rgb = True if x.shape[1] == 3 else False
    if as_rgb:
        # TODO rescaling
        # should undo the standardization from the dataloader
        # multiply by std, then add mean
        fig = po.imshow(
            [curr, target, pred, curr - target, np.NaN*target, pred - target],
            title=['current', 'target', 'prediction', f'mse {copy_mse:.3f}',
                   '', f'mse {mse:.3f}'],
            vrange='auto1', zoom=2, col_wrap=3, as_rgb=as_rgb
        );
    else:
        fig = po.imshow(
            [curr, target, pred, curr - target, np.NaN*target, pred - target],
            title=['current', 'target', 'prediction', f'mse {copy_mse:.3f}',
                   '', f'mse {mse:.3f}'],
            vrange='auto1', zoom=2, col_wrap=3, as_rgb=as_rgb
        );
    # without the error maps
    # fig = po.imshow([target, pred, pred - target],
    #                 title=['target', 'pred', f'mse {mse:.3f}'],
    #                 vrange='auto1', zoom=2, col_wrap=3);
    return fig


def save_filter_viz(
        W, group_size, path=None, freq=True, cw=8, zoom=2, vrange='auto1'
    ):
    """ save filters

    W: [C c 1 H W]
        C = (k g) k groups of size g=2
        c = number of input channels
            (1/3 for grayscale/color images)
    path:
        saves if path is not None
    """
    w = W.data.cpu()[:, :, 0]  # remove the dummy time axis
    Wsorted = sort_by_ave_norm(w, g=group_size).numpy()

    as_rgb = True if w.shape[1] == 3 else False
    if as_rgb:
        fig = po.imshow(
            po.tools.rescale(torch.from_numpy(Wsorted), 0, 1),
            col_wrap=cw, zoom=zoom, title=None, as_rgb=as_rgb,
        );
    else:
        fig = po.imshow(Wsorted, col_wrap=cw, zoom=zoom,
                        title=None, vrange=vrange,);
    if path is not None:
        fig.savefig(path + '.pdf')
    
    # frequency domain: amplitude spectrum
    if freq:
        w_absspec = np.abs(
            np.fft.fftshift(np.fft.fft2(
                Wsorted, norm='ortho', axes=(-2,-1)
            ), axes=(-2,-1))
        )
        if as_rgb:
            # TODO does this rescaling make sense?
            spec = po.imshow(
                po.tools.rescale(torch.from_numpy(w_absspec), 0, 1),
                col_wrap=cw, zoom=zoom, title=None, as_rgb=as_rgb
            );
        else:
            spec = po.imshow(
                torch.from_numpy(w_absspec),
                col_wrap=cw, zoom=zoom, title=None, vrange=vrange
            );
        if path is not None:
            spec.savefig(path + '_absspec.pdf')

        # po.imshow(
        #     w_absspec.mean(0, keepdims=True),
        #     zoom=zoom, vrange=vrange, title="mean of amplitudes"
        # );
    return fig


def save_filter_anim(W, path=None, cw=4, zoom=2, vrange='auto1'):
    """ animate filters by pairs
    
    W: [C c 1 H W]
        C = (k g) k groups of size g=2
        c = number of input channels
            (1/3 for grayscale/color images)
    path:
        saves if path is not none

    Notes
    Only supports paired filters (uninformative if group size > 2)
    check grayscale not broken
    """
    w_in = W.data.cpu()[:, :, 0]  # remove the dummy time axis
    w_in = sort_by_ave_norm(w_in).numpy()

    ko, ki, H, W = w_in.shape
    as_rgb = True if ki == 3 else False
    ww = [np.cos(theta) * w_in[::2] + np.sin(theta) * w_in[1::2]
          for theta in np.linspace(0, 2*np.pi)]
    ww = np.stack(ww, axis=2)  # B C T H W
    if as_rgb:
        ani = po.animshow(
            po.tools.rescale(torch.from_numpy(ww), 0, 1),
            col_wrap=cw, zoom=zoom, title=None, as_rgb=as_rgb, framerate=25
        )
    else:
        ani = po.animshow(
            torch.from_numpy(ww),
            col_wrap=cw, zoom=zoom, title=None, vrange=vrange, framerate=25
        )
    if path is not None:
        ani.save(path + '.gif')

    ww_absspec = np.abs(
        np.fft.fftshift(np.fft.fft2(
            ww, norm='ortho', axes=(-2,-1)
        ), axes=(-2,-1))
    )
    if as_rgb:
        # TODO does this rescaling make sense?
        ani_spec = po.animshow(
            po.tools.rescale(torch.from_numpy(ww_absspec), 0, 1),
                col_wrap=cw, zoom=zoom, title=None, as_rgb=as_rgb, framerate=25
            )
    else:
        ani_spec = po.animshow(
            torch.from_numpy(ww_absspec),
                col_wrap=cw, zoom=zoom, title=None, vrange=vrange, framerate=25
            )
    if path is not None:
        ani_spec.save(path + '_absspec.gif')

    return ani
