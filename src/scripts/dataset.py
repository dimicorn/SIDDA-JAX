import math
from typing import Callable, Optional

import numpy as np


class NpyImageDataset:
    """Dataset backed by a pair of ``.npy`` files: one array of images, one of labels.

    Args:
        input_path (str): Path to the image array (.npy).
        output_path (Optional[str], optional): Path to the label array (.npy). Defaults to None.
        transform (Optional[Callable], optional): Transform applied to each image. Defaults to None.
        target_domain (bool, optional): If True, __getitem__ returns only the image
            (labels, if loaded, are never used during training for the target domain).
    """

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_domain: bool = False,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.transform = transform
        self.target_domain = target_domain

        try:
            self.img = np.load(self.input_path)
            self.label = None
            if not self.target_domain and self.output_path is not None:
                self.label = np.load(self.output_path)
        except Exception as e:
            raise RuntimeError(
                f"Error loading data from {input_path} and {output_path}: {e}"
            )

        if self.img.dtype == np.float64:
            # Some datasets (e.g. gz_evo) ship ~7.7GB float64 arrays per domain --
            # downcast eagerly to halve resident memory. uint8 arrays are left alone so
            # augment.to_float32's dtype-based /255 rescaling still applies correctly.
            self.img = self.img.astype(np.float32)

        if self.img.ndim == 3:
            # (N, H, W) -> (N, H, W, 1), NHWC convention used throughout this pipeline.
            self.img = self.img[..., None]

        if self.label is not None and len(self.img) != len(self.label):
            raise ValueError("Input and output files must have the same length.")

        self.length = len(self.img)

    def __getitem__(self, idx: int):
        img = self.img[idx]
        if self.transform:
            img = self.transform(img)

        if self.target_domain:
            return img

        label = int(self.label[idx])
        return img, label

    def __len__(self) -> int:
        return self.length


class Subset:
    """A view over a subset of a dataset's indices, with its own independent transform.

    Unlike ``torch.utils.data.Subset`` combined with mutating ``dataset.transform`` in
    place (the pattern the original PyTorch code used), this does not touch the parent
    dataset's state at all -- two Subsets built from the same parent can safely carry
    different transforms without clobbering each other.
    """

    def __init__(self, dataset: NpyImageDataset, indices: np.ndarray, transform: Optional[Callable]):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img = self.dataset.img[real_idx]
        if self.transform:
            img = self.transform(img)

        if self.dataset.target_domain:
            return img

        label = int(self.dataset.label[real_idx])
        return img, label


def split_dataset(
    dataset: NpyImageDataset,
    val_size: float,
    seed: int,
    train_transform: Optional[Callable],
    val_transform: Optional[Callable],
):
    """Split a dataset into train/val Subsets, each with its own transform."""
    n = len(dataset)
    n_val = int(n * val_size)
    perm = np.random.default_rng(seed).permutation(n)
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]

    train_subset = Subset(dataset, train_indices, train_transform)
    val_subset = Subset(dataset, val_indices, val_transform)
    return train_subset, val_subset


class NumpyLoader:
    """Minimal batch iterator over a dataset (NpyImageDataset or Subset), yielding
    stacked numpy arrays. Replaces torch.utils.data.DataLoader for this pipeline.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = False,
        seed: Optional[int] = None,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed if seed is not None else 0
        self.drop_last = drop_last
        self._epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            rng = np.random.default_rng((self.seed, self._epoch))
            indices = rng.permutation(n)
            self._epoch += 1
        else:
            indices = np.arange(n)

        target_domain = getattr(self.dataset, "target_domain", None)
        if target_domain is None:
            target_domain = getattr(self.dataset.dataset, "target_domain", False)

        n_batches = len(self)
        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            batch_indices = indices[start : start + self.batch_size]
            if target_domain:
                imgs = np.stack([self.dataset[i] for i in batch_indices])
                yield imgs
            else:
                items = [self.dataset[i] for i in batch_indices]
                imgs = np.stack([it[0] for it in items])
                labels = np.array([it[1] for it in items], dtype=np.int32)
                yield imgs, labels


dataset_dict = {
    "shapes": NpyImageDataset,
    "astro_objects": NpyImageDataset,
    "mnist_m": NpyImageDataset,
    "gz_evo": NpyImageDataset,
    "mrssc2": NpyImageDataset,
}

# shapes_classes matches src/scripts/data/shapes/shapes_dataset/readme.txt ("0: lines,
# 1: rectangles, 2: squares"), confirmed against the actual downloaded dataset -- not
# ("line", "rectangle", "circle") as the original PyTorch code had it.
gz_evo_classes = (
    "barred_spiral",
    "edge_on_disk",
    "featured_without_bar_or_spiral",
    "smooth_cigar",
    "smooth_round",
    "unbarred_spiral",
)
mnist_m_classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
shapes_classes = ("lines", "rectangles", "squares")
astro_objects_classes = ("elliptical", "spiral", "stars")
mrssc2_classes = ("city", "coast", "desert", "farmland", "lake", "mountain", "river")

classes_dict = {
    "shapes": shapes_classes,
    "astro_objects": astro_objects_classes,
    "mnist_m": mnist_m_classes,
    "gz_evo": gz_evo_classes,
    "mrssc2": mrssc2_classes,
}
