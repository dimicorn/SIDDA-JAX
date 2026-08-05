"""Per-item image augmentation, replacing torchvision.transforms for this pipeline.

Images in this repo's datasets are pre-generated ``float64`` NumPy arrays (not uint8
photographs) with values outside [0, 255] in some cases (e.g. the "noise" target-domain
shapes images run up to ~7.0), so PIL (which mostly assumes uint8/[0,255] imagery) is a
poor fit here. scipy.ndimage works directly on arbitrary-range floating point arrays and
handles the (H, W, C) layout used throughout this pipeline uniformly, so it's used for
the geometric transforms (rotation, translation) instead.

Single source of truth for the per-dataset transform construction that used to be
duplicated (with a real divergence bug -- see CLAUDE.md) across four PyTorch scripts.
"""

from typing import Callable

import numpy as np
from scipy import ndimage

_INPUT_SIZES = {
    "shapes": (100, 100),
    "astro_objects": (100, 100),
    "mnist_m": (32, 32),
    "gz_evo": (100, 100),
    "mrssc2": (100, 100),
}


def to_float32(img: np.ndarray) -> np.ndarray:
    """Mirror torchvision.ToTensor's dtype-dependent scaling: uint8 images are
    assumed to be in [0, 255] and rescaled to [0, 1]; any other dtype (this repo's
    datasets are pre-generated float64 arrays, sometimes outside [0, 1]) is just cast.
    """
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def random_rotation(img: np.ndarray, rng: np.random.Generator, max_degrees: float = 180) -> np.ndarray:
    angle = rng.uniform(-max_degrees, max_degrees)
    return ndimage.rotate(img, angle, axes=(0, 1), reshape=False, order=1, mode="constant", cval=0.0)


def random_translate(img: np.ndarray, rng: np.random.Generator, max_frac: float = 0.1) -> np.ndarray:
    h, w = img.shape[0], img.shape[1]
    dy = rng.uniform(-max_frac, max_frac) * h
    dx = rng.uniform(-max_frac, max_frac) * w
    shift = (dy, dx) + (0,) * (img.ndim - 2)
    return ndimage.shift(img, shift=shift, order=1, mode="constant", cval=0.0)


def random_hflip(img: np.ndarray, rng: np.random.Generator, p: float = 0.3) -> np.ndarray:
    if rng.random() < p:
        return img[:, ::-1, ...]
    return img


def random_vflip(img: np.ndarray, rng: np.random.Generator, p: float = 0.3) -> np.ndarray:
    if rng.random() < p:
        return img[::-1, :, ...]
    return img


def resize_if_needed(img: np.ndarray, target_hw: tuple) -> np.ndarray:
    h, w = img.shape[0], img.shape[1]
    if (h, w) == tuple(target_hw):
        return img
    zoom = (target_hw[0] / h, target_hw[1] / w) + (1,) * (img.ndim - 2)
    return ndimage.zoom(img, zoom=zoom, order=1)


def normalize(img: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    return (img - mean) / std


def get_transform(
    dataset_name: str,
    train: bool,
    rng: np.random.Generator,
    input_size: tuple = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a per-item transform function for the given dataset/split.

    Args:
        dataset_name: key into dataset.dataset_dict.
        train: if True, apply random augmentation (rotation/translate/flips) before
            normalizing; if False, only resize + normalize.
        rng: shared numpy Generator the returned closure draws randomness from (advances
            on every call, so successive images get independent augmentations).
        input_size: (H, W) target size; defaults to this dataset's canonical size.
    """
    target_hw = input_size if input_size is not None else _INPUT_SIZES[dataset_name]

    if train:

        def transform(img: np.ndarray) -> np.ndarray:
            img = to_float32(img)
            img = random_rotation(img, rng, max_degrees=180)
            img = random_translate(img, rng, max_frac=0.1)
            img = random_hflip(img, rng, p=0.3)
            img = random_vflip(img, rng, p=0.3)
            img = resize_if_needed(img, target_hw)
            img = normalize(img)
            return img

    else:

        def transform(img: np.ndarray) -> np.ndarray:
            img = to_float32(img)
            img = resize_if_needed(img, target_hw)
            img = normalize(img)
            return img

    return transform
