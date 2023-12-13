import os
from time import time
import argparse
import sys
import warnings
import pkg_resources

from einops import rearrange, reduce, repeat
import numpy as np
import skimage
import torch
from torch import nn
from PIL import Image
import cv2 as cv
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import uniform_filter

from . hipt_model_utils import eval_transforms
from . hipt_4k import HIPT_4K
from typing import Optional, Sequence

Image.MAX_IMAGE_PIXELS = None

def preprocess(img):
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    return img

def rescale(
    image: np.ndarray, scaling_factor: float, resample: int = Image.NEAREST
) -> np.ndarray:
    r'''
    Rescales image by a given `scaling_factor`

    :param image: Image array
    :param scaling_factor: Scaling factor
    :param resample: Resampling filter
    :returns: The rescaled image
    '''
    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize(
        [round(x * scaling_factor) for x in image_pil.size], resample=resample,
    )
    return np.array(image_pil)

def remove_fg_elements(mask: np.ndarray, size_threshold: float):
    r'''Removes small foreground elements'''
    labels, _ = label(mask)
    labels_unique, label_counts = np.unique(labels, return_counts=True)
    small_labels = labels_unique[
        label_counts < size_threshold ** 2 * np.prod(mask.shape)
    ]
    mask[np.isin(labels, small_labels)] = False
    return mask

def cleanup_mask(mask: np.ndarray, size_threshold: float):
    r'''Removes small background and foreground elements'''
    mask = ~remove_fg_elements(~mask, size_threshold)
    mask = remove_fg_elements(mask, size_threshold)
    return mask

def resize(
    image: np.ndarray,
    target_shape: Sequence[int],
    resample: int = Image.NEAREST,
) -> np.ndarray:
    r'''
    Resizes image to a given `target_shape`

    :param image: Image array
    :param target_shape: Target shape
    :param resample: Resampling filter
    :returns: The rescaled image
    '''
    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize(target_shape[::-1], resample=resample)
    return np.array(image_pil)


def compute_tissue_mask(
    image: np.ndarray,
    convergence_threshold: float = 0.0001,
    size_threshold: float = 0.01,
    initial_mask: Optional[np.ndarray] = None,
    max_iter: int = 100,
) -> np.ndarray:
    r'''
    Computes boolean mask indicating likely foreground elements in histology
    image.
    '''
    # pylint: disable=no-member
    # ^ pylint fails to identify cv.* members
    original_shape = image.shape[:2]
    scale_factor = 1000 / max(original_shape)
    image = rescale(image, scale_factor, resample=Image.NEAREST)
    image = cv.blur(image, (5, 5))
    if initial_mask is None:
        initial_mask = (
            cv.blur(cv.Canny(image, 100, 200), (20, 20)) > 0
        )
    else:
        initial_mask = rescale(
                initial_mask, scale_factor, resample=Image.NEAREST)
    initial_mask = binary_fill_holes(initial_mask)
    initial_mask = remove_fg_elements(initial_mask, 0.1)  # type: ignore
    mask = np.where(initial_mask, cv.GC_PR_FGD, cv.GC_PR_BGD)
    mask = mask.astype(np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = bgd_model.copy()
    print('Computing tissue mask:')
    for i in range(max_iter):
        old_mask = mask.copy()
        try:
            cv.grabCut(
                image,
                mask,
                None,
                bgd_model,
                fgd_model,
                1,
                cv.GC_INIT_WITH_MASK,
            )
        except cv.error as cv_err:
            warnings.warn(f'Failed to mask tissue\n{str(cv_err).strip()}')
            mask = np.full_like(mask, cv.GC_PR_FGD)
            break
        prop_changed = (mask != old_mask).sum() / np.prod(mask.shape)
        print('  Iteration %2d Δ = %.2f%%', i, 100 * prop_changed)
        if prop_changed < convergence_threshold:
            break
    mask = np.isin(mask, [cv.GC_FGD, cv.GC_PR_FGD])
    mask = cleanup_mask(mask, size_threshold)
    mask = resize(mask, target_shape=original_shape, resample=Image.NEAREST)
    return mask


def remove_border(x):
    x = x.copy()
    x[0] = 0
    x[-1] = 0
    x[:, 0] = 0
    x[:, -1] = 0
    return x


def get_extent(mask):
    extent = []
    for ax in range(mask.ndim):
        ma = mask.swapaxes(0, ax)
        ma = ma.reshape(ma.shape[0], -1)
        notempty = ma.any(1)
        start = notempty.argmax()
        stop = notempty.size - notempty[::-1].argmax()
        extent.append([start, stop])
    extent = np.array(extent)
    return extent

def crop_image(img, extent):
    extent = np.array(extent)
    pad = np.zeros((img.ndim, 2), dtype=int)
    for i, (lower, upper) in enumerate(extent):
        if lower < 0:
            pad[i][0] = 0 - lower
        if upper > img.shape[i]:
            pad[i][1] = upper - img.shape[i]
    if (pad != 0).any():
        img = np.pad(img, pad, mode='edge')
        extent += pad[:extent.shape[0], [0]]
    for i, (lower, upper) in enumerate(extent):
        img = img.take(range(lower, upper), axis=i)
    return img

def adjust_margins(img, mask, pad=0):
    print('Padding margins...')
    extent = np.stack([[0, 0], mask.shape]).T
    # make size divisible by pad without changing coords
    extent[:, 1] += pad*2
    extent[:, 1] -= (extent[:, 1] - extent[:, 0]) % pad
    img = crop_image(img, extent)
    mask = crop_image(mask[..., np.newaxis], extent)[..., 0]
    return img, mask, extent

def shrink_mask(x, size):
    size = size * 2 - 1
    x = uniform_filter(x.astype(float), size=size)
    x = np.isclose(x, 1)
    return x


def patchify(x, patch_size):
    shape_ori = np.array(x.shape[:2])
    shape_ext = (
            (shape_ori + patch_size - 1)
            // patch_size * patch_size)
    x = np.pad(
            x,
            (
                (0, shape_ext[0] - x.shape[0]),
                (0, shape_ext[1] - x.shape[1]),
                (0, 0)),
            mode='edge')
    tiles_shape = np.array(x.shape[:2]) // patch_size
    # x = rearrange(
    #         x, '(h1 h) (w1 w) c -> h1 w1 h w c',
    #         h=patch_size, w=patch_size)
    # x = rearrange(
    #         x, '(h1 h) (w1 w) c -> (h1 w1) h w c',
    #         h=patch_size, w=patch_size)
    tiles = []
    for i0 in range(tiles_shape[0]):
        a0 = i0 * patch_size  # TODO: change to patch_size[0]
        b0 = a0 + patch_size  # TODO: change to patch_size[0]
        for i1 in range(tiles_shape[1]):
            a1 = i1 * patch_size  # TODO: change to patch_size[1]
            b1 = a1 + patch_size  # TODO: change to patch_size[1]
            tiles.append(x[a0:b0, a1:b1])
    shapes = dict(
            original=shape_ori,
            padded=shape_ext,
            tiles=tiles_shape)
    return tiles, shapes


def get_embeddings_sub(model, x):
    mask = x[..., -1] > 0  # get foreground mask
    x = x[..., :-1]
    x = x.astype(np.float32) / 255.0
    x = eval_transforms()(x)
    x_cls, x_sub = model.forward_all256(x[None])
    x_cls = x_cls.cpu().detach().numpy()
    x_sub = x_sub.cpu().detach().numpy()
    x_cls = x_cls[0].transpose(1, 2, 0)
    x_sub = x_sub[0].transpose(1, 2, 3, 4, 0)
    m_sub = reduce(
            mask,
            '(h2 h3 h) (w2 w3 w) -> h2 w2 h3 w3', 'max',
            h2=x_sub.shape[0], w2=x_sub.shape[1],
            h3=x_sub.shape[2], w3=x_sub.shape[3])
    x_sub[~m_sub] = np.nan
    return x_cls, x_sub


def get_embeddings_cls(model, x):
    x = torch.tensor(x.transpose(2, 0, 1))
    with torch.no_grad():
        __, x_sub4k = model.forward_all4k(x[None])
    x_sub4k = x_sub4k.cpu().detach().numpy()
    x_sub4k = x_sub4k[0].transpose(1, 2, 0)
    return x_sub4k


def get_embeddings(img, pretrained=True, device='cuda'):
    '''
    Extract embeddings from histology tiles
    Args:
        tiles: Histology image tiles.
            Shape: (N, H, W, C).
            `H` and `W` are both divisible by 256.
            Channels `C` include R, G, B, foreground mask.
    Returns:
        emb_cls: Embeddings of (256 x 256)-sized patches
            Shape: (H/256, W/256, 384)
        emb_sub: Embeddings of (16 x 16)-sized patches
            Shape: (H/16, W/16, 384)
    '''
    print('Extracting embeddings...')
    t0 = time()

    tile_size = 4096
    tiles, shapes = patchify(img, patch_size=tile_size)
    
    model256_path, model4k_path = None, None
    if pretrained:
        model256_path = pkg_resources.resource_filename(__name__,'checkpoints/vit256_small_dino.pth')
        model4k_path = pkg_resources.resource_filename(__name__,'checkpoints/vit4k_xs_dino.pth')
    model = HIPT_4K(
            model256_path=model256_path,
            model4k_path=model4k_path,
            device256=device, device4k=device)
    model.eval()
    patch_size = (256, 256)
    subpatch_size = (16, 16)
    n_subpatches = tuple(
            a // b for a, b in zip(patch_size, subpatch_size))

    emb_sub = []
    emb_mid = []
    for i in range(len(tiles)):
        if i % 10 == 0:
            print('tile', i, '/', len(tiles))
        x_mid, x_sub = get_embeddings_sub(model, tiles[i])
        emb_mid.append(x_mid)
        emb_sub.append(x_sub)
    del tiles
    torch.cuda.empty_cache()
    emb_mid = rearrange(
            emb_mid, '(h1 w1) h2 w2 k -> (h1 h2) (w1 w2) k',
            h1=shapes['tiles'][0], w1=shapes['tiles'][1])

    emb_cls = get_embeddings_cls(model, emb_mid)
    del emb_mid, model
    torch.cuda.empty_cache()

    shape_orig = np.array(shapes['original']) // subpatch_size

    chans_sub = []
    for i in range(emb_sub[0].shape[-1]):
        chan = rearrange(
                np.array([e[..., i] for e in emb_sub]),
                '(h1 w1) h2 w2 h3 w3 -> (h1 h2 h3) (w1 w2 w3)',
                h1=shapes['tiles'][0], w1=shapes['tiles'][1])
        chan = chan[:shape_orig[0], :shape_orig[1]]
        chans_sub.append(chan)
    del emb_sub

    mask = np.isfinite(chans_sub[0])
    chans_cls = []
    for i in range(emb_cls[0].shape[-1]):
        chan = repeat(
                np.array([e[..., i] for e in emb_cls]),
                'h12 w12 -> (h12 h3) (w12 w3)',
                h3=n_subpatches[0], w3=n_subpatches[1])
        chan = chan[:shape_orig[0], :shape_orig[1]]
        chan[~mask] = np.nan
        chans_cls.append(chan)
    del emb_cls

    print(int(time() - t0), 'sec')

    return chans_cls, chans_sub


def get_embeddings_shift(img, pretrained=True, device='cuda'):
    factor = 16  # scaling factor between cls and sub. Fixed
    margin = 256  # margin for shifting. Divisble by 256
    stride = 64  # stride for shifting. Divides `margin`.
    shape_emb = np.array(img.shape[:2]) // factor
    chans_cls = [
            np.zeros(shape_emb, dtype=np.float32)
            for __ in range(192)]
    chans_sub = [
            np.zeros(shape_emb, dtype=np.float32)
            for __ in range(384)]
    start_list = list(range(0, margin, stride))
    n_reps = len(start_list)**2
    for start in start_list:
        start0, start1 = start, start
        print(f'shift {start0}/{margin}, {start1}/{margin}')
        t0 = time()
        stop0, stop1 = -margin+start0, -margin+start1
        im = img[start0:stop0, start1:stop1]
        cls, sub = get_embeddings(
                im, pretrained=pretrained, device=device)
        del im
        sta0, sta1 = start0 // factor, start1 // factor
        sto0, sto1 = stop0 // factor, stop1 // factor
        for i in range(len(chans_cls)):
            chans_cls[i][sta0:sto0, sta1:sto1] += cls[i]
        del cls
        for i in range(len(chans_sub)):
            chans_sub[i][sta0:sto0, sta1:sto1] += sub[i]
        del sub
        print(int(time() - t0), 'sec')

    mar = margin // factor
    for chan in chans_cls:
        chan /= n_reps
        chan[-mar:] = np.nan
        chan[:, -mar:] = np.nan
    for chan in chans_sub:
        chan /= n_reps
        chan[-mar:] = np.nan
        chan[:, -mar:] = np.nan

    return chans_cls, chans_sub


def impute_missing(x, mask, radius=3, method='ns'):
    method_dict = {
            'telea': cv.INPAINT_TELEA,
            'ns': cv.INPAINT_NS}
    method = method_dict[method]
    channels = [x[..., i] for i in range(x.shape[-1])]
    mask = mask.astype(np.uint8)
    y = [cv.inpaint(c, mask, radius, method) for c in channels]
    y = np.stack(y, -1)
    return y


def smoothen(
        x, size, method='cv', mode='mean',
        fill_missing=False, device='cuda'):
    mask = np.isfinite(x).all(-1)
    x = impute_missing(x, ~mask)
    if method == 'gf':
        y = skimage.filters.gaussian(
                x, sigma=size, preserve_range=True, channel_axis=-1)
    elif method == 'cv':
        kernel = np.ones((size, size), np.float32) / size**2
        y = cv.filter2D(
                x, ddepth=-1, kernel=kernel, borderType=cv.BORDER_REFLECT)
        if y.ndim == 2:
            y = y[..., np.newaxis]
    elif method == 'cnn':
        assert isinstance(size, int)
        padding = size // 2
        size = size + 1

        pool_dict = {
                'mean': nn.AvgPool2d(
                    kernel_size=size, stride=1, padding=0),
                'max': nn.MaxPool2d(
                    kernel_size=size, stride=1, padding=0)}
        pool = pool_dict[mode]

        mod = nn.Sequential(
                nn.ReflectionPad2d(padding),
                pool)
        y = mod(torch.tensor(x, device=device).permute(2, 0, 1))
        y = y.permute(1, 2, 0)
        y = y.cpu().detach().numpy()
    else:
        raise ValueError(f'Method `{method}` not recognized')
    if not fill_missing:
        y[~mask] = np.nan
    return y


def smoothen_embeddings(embs, size, method='cnn', groups=None, device='cuda'):
    if groups is None:
        groups = embs.keys()
    out = {}
    for grp, em in embs.items():
        if grp in groups:
            if isinstance(em, list):
                smoothened = [
                        smoothen(
                            c[..., np.newaxis], size, method,
                            device=device)[..., 0]
                        for c in em]
            else:
                smoothened = smoothen(em, size, method, device=device)
        else:
            smoothened = em
        out[grp] = smoothened
    return out


def match_foregrounds(embs):
    print('Matching foregrounds...')
    t0 = time()
    channels = np.concatenate(list(embs.values()))
    mask = np.isfinite(channels).all(0)
    for group, channels in embs.items():
        for chan in channels:
            chan[~mask] = np.nan
    print(int(time() - t0), 'sec')


def adjust_weights(embs, weights=None):
    print('Adjusting weights...')
    t0 = time()
    if weights is None:
        weights = {grp: 1.0 for grp in embs.keys()}
    for grp in embs.keys():
        channels = embs[grp]
        wt = weights[grp]
        means = np.array([np.nanmean(chan) for chan in channels])
        std = np.sum([np.nanvar(chan) for chan in channels])**0.5
        for chan, me in zip(channels, means):
            chan[:] -= me
            chan[:] /= std
            chan[:] *= wt**0.5
    print(int(time() - t0), 'sec')



def get_features(img,locs,rad,pretrained=True,device='cpu'):
  img = preprocess(img)
  mask = compute_tissue_mask(img)
  mask = remove_border(mask)
  img, mask, __ = adjust_margins(img, mask, pad=256)
  img[~mask] = 0
  mask = shrink_mask(mask, size=256)
  mask = mask[..., np.newaxis].astype(np.uint8) * 255
  img = np.concatenate([img, mask], -1)
  emb_cls, emb_sub = get_embeddings_shift(img, pretrained=True, device=device)
  embs = dict(cls=emb_cls, sub=emb_sub)
  embs = smoothen_embeddings(embs, size=16, groups=['cls'],method='cv',fill_missing=True,device=device)
  match_foregrounds(embs)
  adjust_weights(embs)  # use uniform weights by default
  cls1 = rearrange(emb_cls, 'c h w -> h w c')
  sub1 = rearrange(emb_sub, 'c h w -> h w c')
  cls_sub1 = np.concatenate((cls1, sub1), 2)
  cls_sub2 = np.stack([rearrange(cls_sub1[(int(np.ceil((locs['4'][i]-rad)/16))):(int(np.floor((locs['4'][i]+rad)/16))),(int(np.ceil((locs['5'][i]-rad)/16))):(int(np.floor((locs['5'][i]+rad)/16)))], 'h w c -> (h w) c').mean(0) for i in range(locs.shape[0])])
  return cls_sub2
