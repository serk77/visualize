import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
import numpy as np

matplotlib.use("Agg")

from visualize import visualize_mri_cases


class VisualizeMriCasesTests(unittest.TestCase):
    def setUp(self):
        self.figure_patcher = patch("matplotlib.pyplot.show")
        self.figure_patcher.start()

    def tearDown(self):
        self.figure_patcher.stop()
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_folder_displays_nifti_files_and_clamps_slice_index(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            (folder / "scan_flair.nii.gz").touch()
            (folder / "scan_seg.nii").touch()
            (folder / "notes.txt").touch()
            data = np.zeros((4, 5, 6))

            with patch(
                "visualize.mri.nib.load",
                return_value=SimpleNamespace(get_fdata=lambda: data),
            ) as load:
                figure, axes = visualize_mri_cases(folder, slice_index=20)

            self.assertEqual(load.call_count, 2)
            self.assertEqual(len(axes), 2)
            self.assertEqual(axes[0].get_title(), "FLAIR\n(Slice 5)")
            self.assertEqual(axes[1].get_title(), "SEG\n(Slice 5)")
            self.assertEqual(axes[1].images[0].get_cmap().name, "tab10")
            self.assertEqual(len(figure.axes), 2)

    def test_list_warns_and_skips_non_nifti_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "scan_t1.nii.gz"
            image_path.touch()
            data = np.zeros((3, 3, 3))

            with patch(
                "visualize.mri.nib.load",
                return_value=SimpleNamespace(get_fdata=lambda: data),
            ):
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    _, axes = visualize_mri_cases(
                        [image_path, "ignored.txt"], slice_index=1
                    )

            self.assertEqual(len(axes), 1)
            self.assertIn("not a NIfTI image", str(caught_warnings[0].message))

    def test_rejects_non_3d_volumes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "scan.nii"
            image_path.touch()
            data = np.zeros((3, 3, 3, 2))

            with patch(
                "visualize.mri.nib.load",
                return_value=SimpleNamespace(get_fdata=lambda: data),
            ):
                with self.assertRaisesRegex(ValueError, "Expected a 3D"):
                    visualize_mri_cases(image_path)

    def test_raises_when_no_nifti_files_are_found(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "No NIfTI files"):
                visualize_mri_cases(temporary_directory)


if __name__ == "__main__":
    unittest.main()
