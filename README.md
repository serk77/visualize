# visualize

## Installation

Install the package and its dependencies in editable mode while developing:

```bash
python -m pip install -e .
```

## Quick start

```python
from visualize import visualize_mri_cases

# Display the middle axial slice of every NIfTI image in a folder.
visualize_mri_cases("path/to/mri_cases")

# Or display a specific slice from selected files.
visualize_mri_cases(
    ["path/to/patient_flair.nii.gz", "path/to/patient_seg.nii.gz"],
    slice_index=80,
)
```

The initial implementation supports three-dimensional `.nii` and `.nii.gz`
volumes and returns the Matplotlib figure and axes after displaying them.

## Initial functionality

The first version will focus on making medical images and deep-learning
predictions easy to inspect in Python:

1. **Load image data** from NumPy arrays and NIfTI files, retaining voxel
   spacing and orientation metadata where available.
2. **Browse 3D volumes** as axial, coronal, and sagittal slices, with slice
   navigation and readable orientation labels.
3. **Display medical images** with grayscale and configurable window/level
   controls.
4. **Overlay segmentations** and label maps on images, with configurable colors
   and transparency.
5. **Work with deep-learning data** by accepting NumPy arrays and PyTorch
   tensors, including common channel-first layouts.
6. **Compare results** by viewing an input image alongside a prediction and,
   when available, a ground-truth mask.
7. **Export views** as image files for reports, experiment notes, and
   debugging.

After these basics, we can consider DICOM series, interactive 3D rendering,
and additional image formats based on project needs.