"""Visualization helpers for MRI data."""

import os
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.figure import Figure

PathInput = Union[str, os.PathLike]
InputSource = Union[PathInput, Sequence[PathInput]]


def _is_nifti(path: Path) -> bool:
    """Return whether a path has a supported NIfTI extension."""
    return path.name.lower().endswith((".nii", ".nii.gz"))


def _resolve_file_paths(input_source: InputSource) -> list[Path]:
    if isinstance(input_source, (str, os.PathLike)):
        source = Path(input_source)
        if source.is_dir():
            return sorted(
                path for path in source.iterdir() if path.is_file() and _is_nifti(path)
            )
        if source.is_file():
            if _is_nifti(source):
                return [source]
            warnings.warn(
                f"Skipping file (not a NIfTI image): {source.name}", UserWarning
            )
            return []
        raise FileNotFoundError(f"Input path does not exist: {source}")

    if not isinstance(input_source, Sequence):
        raise TypeError("input_source must be a folder/file path or a sequence of paths.")

    file_paths = []
    for item in input_source:
        if not isinstance(item, (str, os.PathLike)):
            raise TypeError("Every item in input_source must be a file path.")
        path = Path(item)
        if not _is_nifti(path):
            warnings.warn(
                f"Skipping file (not a NIfTI image): {path.name}", UserWarning
            )
            continue
        file_paths.append(path)

    return sorted(file_paths)


def _image_title(path: Path) -> str:
    filename = path.name.lower()
    for modality in ("flair", "t1ce", "t1", "t2", "seg"):
        if modality in filename:
            return modality.upper()
    return path.name.removesuffix(".nii.gz").removesuffix(".nii")


def visualize_mri_cases(
    input_source: InputSource, slice_index: Optional[int] = None
) -> tuple[Figure, np.ndarray]:
    """Display axial slices from NIfTI MRI files side by side.

    ``input_source`` may be a folder path, a single file path, or a sequence
    of file paths. Folder inputs include NIfTI files in that folder only.
    Three-dimensional volumes are supported; each image uses its own middle
    slice when ``slice_index`` is omitted.

    Parameters
    ----------
    input_source:
        Folder path, file path, or sequence of file paths to NIfTI images.
    slice_index:
        Zero-based index along the third volume axis. Values outside an
        image's range are clamped to its nearest valid slice.

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray]
        The figure and one-dimensional array of plot axes.
    """
    if slice_index is not None and (
        isinstance(slice_index, bool) or not isinstance(slice_index, int)
    ):
        raise TypeError("slice_index must be an integer or None.")

    file_paths = _resolve_file_paths(input_source)
    if not file_paths:
        raise ValueError("No NIfTI files found to display.")

    valid_images = [(_image_title(path), path) for path in file_paths]
    slices = []
    for title, path in valid_images:
        image = nib.load(str(path))
        data = image.get_fdata()
        if data.ndim != 3:
            raise ValueError(
                f"Expected a 3D NIfTI volume, got {data.ndim}D data: {path}"
            )

        target_slice = (
            data.shape[2] // 2
            if slice_index is None
            else min(max(0, slice_index), data.shape[2] - 1)
        )
        slices.append((title, target_slice, np.rot90(data[:, :, target_slice])))

    fig, axes_grid = plt.subplots(
        1, len(slices), figsize=(4 * len(slices), 5), squeeze=False
    )
    axes = axes_grid[0]
    for ax, (title, target_slice, slice_data) in zip(axes, slices):
        if title == "SEG":
            ax.imshow(slice_data, cmap="tab10", interpolation="nearest")
        else:
            ax.imshow(slice_data, cmap="gray")

        ax.set_title(f"{title}\n(Slice {target_slice})")
        ax.axis("off")

    fig.tight_layout()
    plt.show()
    return fig, axes
