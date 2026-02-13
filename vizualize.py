import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


#lf vs hf volumes 112 × 138 × 40	179 × 221 × 200

project_root = os.path.dirname(os.path.abspath(__file__))
lf_folder = os.path.join(project_root, "mri_resolution", "train", "low_field")
hf_folder = os.path.join(project_root, "mri_resolution", "train", "high_field")


def load_nii_volume(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    zooms = nib.affines.voxel_sizes(img.affine)
    return data, zooms


def resample_to_target(source, source_zooms, target_shape, target_zooms):
    # Resample using voxel spacing to preserve physical size.
    new_shape = np.round(  # Compute the target voxel grid size in index space.
        np.array(source.shape)  # Convert source shape to a numeric array.
        * np.array(source_zooms)  # Scale by source voxel sizes (mm/voxel).
        / np.array(target_zooms)  # Divide by target voxel sizes (mm/voxel).
    ).astype(int)  # Convert to integer voxel counts for the new grid.
    zoom_factors = new_shape / np.array(source.shape)  # Per-axis scaling factors.
    resampled = zoom(source, zoom_factors, order=1)  # Linear interpolation.
    # Pad or crop to exact target shape.
    result = np.zeros(target_shape, dtype=resampled.dtype)  # Allocate target array.
    min_shape = np.minimum(result.shape, resampled.shape)  # Overlap region size.
    result[: min_shape[0], : min_shape[1], : min_shape[2]] = resampled[
        : min_shape[0], : min_shape[1], : min_shape[2]
    ]  # Copy the overlapping region into the target array.
    return result  # Return the resampled volume with the requested shape.

def vizualize_lf_and_hf(file_number, slice_index=0):
    lf_files = sorted(os.listdir(lf_folder))
    hf_files = sorted(os.listdir(hf_folder))

    lf_path = os.path.join(lf_folder, lf_files[file_number])
    hf_path = os.path.join(hf_folder, hf_files[file_number])

    lf_data, lf_zooms = load_nii_volume(lf_path)
    hf_data, hf_zooms = load_nii_volume(hf_path)
    hf_resampled = resample_to_target(hf_data, hf_zooms, lf_data.shape, lf_zooms)
    slice_index = max(0, min(slice_index, lf_data.shape[2] - 1))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Low Field MRI")
    plt.imshow(lf_data[:, :, slice_index], cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("High Field MRI")
    plt.imshow(hf_resampled[:, :, slice_index], cmap='gray')
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    vizualize_lf_and_hf(file_number=4, slice_index=20)
