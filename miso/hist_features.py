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
from skimage.transform import rescale

from . hipt_model_utils import eval_transforms
from . hipt_4k import HIPT_4K
from typing import Optional, Sequence

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None


def rescale_image(img, scale):
    img = np.array(img).astype(np.float32)
    if img.ndim == 2:
        scale = [scale, scale]
    elif img.ndim == 3:
        scale = [scale, scale, 1]
    else:
        raise ValueError('Unrecognized image ndim')
    img = rescale(img, scale, preserve_range=True)
    img = img.astype(np.uint8)
    return img

def preprocess(img):
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    return img

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


def crop_image(img, extent, mode='edge', constant_values=None):
    extent = np.array(extent)
    pad = np.zeros((img.ndim, 2), dtype=int)
    for i, (lower, upper) in enumerate(extent):
        if lower < 0:
            pad[i][0] = 0 - lower
        if upper > img.shape[i]:
            pad[i][1] = upper - img.shape[i]
    if (pad != 0).any():
        kwargs = {}
        if mode == 'constant' and constant_values is not None:
            kwargs['constant_values'] = constant_values
        img = np.pad(img, pad, mode=mode, **kwargs)
        extent += pad[:extent.shape[0], [0]]
    for i, (lower, upper) in enumerate(extent):
        img = img.take(range(lower, upper), axis=i)
    return img



def adjust_margins(img, pad, pad_value=None):
    extent = np.stack([[0, 0], img.shape[:2]]).T
    # make size divisible by pad without changing coords
    remainder = (extent[:, 1] - extent[:, 0]) % pad
    complement = (pad - remainder) % pad
    extent[:, 1] += complement
    if pad_value is None:
        mode = 'edge'
    else:
        mode = 'constant'
    img = crop_image(
            img, extent, mode=mode, constant_values=pad_value)
    return img


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
    x = x.astype(np.float32) / 255.0
    x = eval_transforms()(x)
    x_cls, x_sub = model.forward_all256(x[None])
    x_cls = x_cls.cpu().detach().numpy()
    x_sub = x_sub.cpu().detach().numpy()
    x_cls = x_cls[0].transpose(1, 2, 0)
    x_sub = x_sub[0].transpose(1, 2, 3, 4, 0)
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
    #print('Extracting embeddings...')
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
        #if i % 10 == 0:
        #    print('tile', i, '/', len(tiles))
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

    chans_cls = []
    for i in range(emb_cls[0].shape[-1]):
        chan = repeat(
                np.array([e[..., i] for e in emb_cls]),
                'h12 w12 -> (h12 h3) (w12 w3)',
                h3=n_subpatches[0], w3=n_subpatches[1])
        chan = chan[:shape_orig[0], :shape_orig[1]]
        chans_cls.append(chan)
    del emb_cls

    #print(int(time() - t0), 'sec')

    return chans_cls, chans_sub


def get_embeddings_shift(
        img, margin=256, stride=64,
        pretrained=True, device='cuda'):
    # margin: margin for shifting. Divisble by 256
    # stride: stride for shifting. Divides `margin`.
    factor = 16  # scaling factor between cls and sub. Fixed
    shape_emb = np.array(img.shape[:2]) // factor
    chans_cls = [
            np.zeros(shape_emb, dtype=np.float32)
            for __ in range(192)]
    chans_sub = [
            np.zeros(shape_emb, dtype=np.float32)
            for __ in range(384)]
    start_list = list(range(0, margin, stride))
    n_reps = 0
    for k,start0 in enumerate(start_list):
        for start1 in tqdm(start_list, desc = 'Extracting image features: ' + str(k+1) + '/' + str(len(start_list))):
            #print(f'shift {start0}/{margin}, {start1}/{margin}')
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
            n_reps += 1
            #print(int(time() - t0), 'sec')

    mar = margin // factor
    for chan in chans_cls:
        chan /= n_reps
        chan[-mar:] = 0.0
        chan[:, -mar:] = 0.0
    for chan in chans_sub:
        chan /= n_reps
        chan[-mar:] = 0.0
        chan[:, -mar:] = 0.0

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
        x, size, kernel='gaussian', backend='cv', mode='mean',
        impute_missing_values=True, device='cuda'):

    if x.ndim == 3:
        expand_dim = False
    elif x.ndim == 2:
        expand_dim = True
        x = x[..., np.newaxis]
    else:
        raise ValueError('ndim must be 2 or 3')

    mask = np.isfinite(x).all(-1)
    if (~mask).any() and impute_missing_values:
        x = impute_missing(x, ~mask)

    if kernel == 'gaussian':
        sigma = size / 4  # approximate std of uniform filter 1/sqrt(12)
        truncate = 4.0
        winsize = np.ceil(sigma * truncate).astype(int) * 2 + 1
        if backend == 'cv':
            print(f'gaussian filter: winsize={winsize}, sigma={sigma}')
            y = cv.GaussianBlur(
                    x, (winsize, winsize), sigmaX=sigma, sigmaY=sigma,
                    borderType=cv.BORDER_REFLECT)
        elif backend == 'skimage':
            y = skimage.filters.gaussian(
                    x, sigma=sigma, truncate=truncate,
                    preserve_range=True, channel_axis=-1)
        else:
            raise ValueError('backend must be cv or skimage')
    elif kernel == 'uniform':
        if backend == 'cv':
            kernel = np.ones((size, size), np.float32) / size**2
            y = cv.filter2D(
                    x, ddepth=-1, kernel=kernel,
                    borderType=cv.BORDER_REFLECT)
            if y.ndim == 2:
                y = y[..., np.newaxis]
        elif backend == 'torch':
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
            raise ValueError('backend must be cv or torch')
    else:
        raise ValueError('kernel must be gaussian or uniform')

    if not mask.all():
        y[~mask] = np.nan

    if expand_dim and y.ndim == 3:
        y = y[..., 0]

    return y


def smoothen_embeddings(
        embs, size, kernel,
        method='cv', groups=None, device='cuda'):
    if groups is None:
        groups = embs.keys()
    out = {}
    for grp, em in embs.items():
        if grp in groups:
            if isinstance(em, list):
                smoothened = [
                        smoothen(
                            c[..., np.newaxis], size=size,
                            kernel=kernel, backend=method,
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



def get_features(img,locs,rad,pixel_size_raw,pixel_size=0.5,pretrained=True,device='cpu'):
  scale = pixel_size_raw / pixel_size
  print('Scaling image')
  img = rescale_image(img, scale = scale)
  rad = rad*scale
  locs1 = locs.copy()
  locs1['4'] = locs1['4']*scale
  locs1['5'] = locs1['5']*scale
  print('Preprocessing image')
  img = preprocess(img)
  #mask = compute_tissue_mask(img)
  #mask = remove_border(mask)
  print('Adjusting margins')
  img = adjust_margins(img, pad=256, pad_value=255)
  #img[~mask] = 0
  #mask = shrink_mask(mask, size=256)
  #mask = mask[..., np.newaxis].astype(np.uint8) * 255
  #img = np.concatenate([img, mask], -1)
  #print('Extracting image features')
  emb_cls, emb_sub = get_embeddings_shift(img, pretrained=True, device=device)
  embs = dict(cls=emb_cls, sub=emb_sub)
  print('Smoothing embeddings')
  embs = smoothen_embeddings(embs, size=16, kernel='uniform', groups=['cls'], method='cv', device=device)
  embs = smoothen_embeddings(embs, size=4, kernel='uniform', groups=['sub'], method='cv', device=device)
  #match_foregrounds(embs)
  #adjust_weights(embs)  # use uniform weights by default
  cls1 = rearrange(emb_cls, 'c h w -> h w c')
  sub1 = rearrange(emb_sub, 'c h w -> h w c')
  cls_sub1 = np.concatenate((cls1, sub1), 2)
  cls_sub2 = np.stack([rearrange(cls_sub1[(int(np.ceil((locs1['4'][i]-rad)/16))):(int(np.floor((locs1['4'][i]+rad)/16))),(int(np.ceil((locs1['5'][i]-rad)/16))):(int(np.floor((locs1['5'][i]+rad)/16)))], 'h w c -> (h w) c').mean(0) for i in range(locs1.shape[0])])
  return cls_sub2
