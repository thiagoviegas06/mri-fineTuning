import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import zoom

def resize_bicubic(source_lf, target_shape_lf):
    """
    Resize a 3D volume to the target shape using bicubic interpolation.

    Args:
        source: 3D numpy array (x, y, z)
        target_shape: Tuple of (x, y, z) for the desired output shape

    Returns:
        Resized 3D numpy array
    """
    lf = nib.load(source_lf)
    source = lf.get_fdata(dtype=np.float32)
    zoom_factors = np.array(target_shape_lf) / np.array(source.shape)
    return zoom(source, zoom_factors, order=3)


def normalize_volume(volume):
    """
    Normalize a 3D volume to the range [0, 1].

    Args:
        volume: 3D numpy array

    Returns:
        Normalized 3D numpy array
    """
    min_val = np.min(volume)
    max_val = np.max(volume)
    if max_val - min_val > 0:
        return (volume - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(volume)
    
"Turning 3D volumes into patches for training the model."
def patch_volume(volume):
    patch_size = 32
    stride = 16
    patches = []
    for x in range(0, volume.shape[0] - patch_size + 1, stride):
        for y in range(0, volume.shape[1] - patch_size + 1, stride):
            for z in range(0, volume.shape[2] - patch_size + 1, stride):
                patch = volume[x:x+patch_size, y:y+patch_size, z:z+patch_size]
                patches.append(patch)
    return np.array(patches)


def _load_canonical(path):
    img = nib.load(path)
    return nib.as_closest_canonical(img)


def _resample_to_reference(moving_img, reference_img, order=1):
    return resample_from_to(moving_img, reference_img, order=order)


def _rigid_register_sitk(moving_img, fixed_img, debug=False):
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required for rigid registration. Install with pip install SimpleITK."
        ) from exc

    def _sitk_from_nifti(img):
        data = img.get_fdata(dtype=np.float32)
        sitk_img = sitk.GetImageFromArray(data)
        affine = img.affine
        spacing = nib.affines.voxel_sizes(affine)
        origin = affine[:3, 3]
        direction = affine[:3, :3] / spacing
        sitk_img.SetSpacing(tuple(spacing))
        sitk_img.SetOrigin(tuple(origin))
        sitk_img.SetDirection(tuple(direction.flatten()))
        return sitk_img

    if debug:
        print("[registration] building SimpleITK images")
    moving = _sitk_from_nifti(moving_img)
    fixed = _sitk_from_nifti(fixed_img)

    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    if debug:
        print("[registration] configuring registration")
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.02)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-4, 200)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial, inPlace=False)

    if debug:
        print("[registration] running registration")
    final = reg.Execute(fixed, moving)
    if debug:
        print("[registration] resampling")
    resampled = sitk.Resample(
        moving,
        fixed,
        final,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )
    resampled_img = nib.Nifti1Image(
        sitk.GetArrayFromImage(resampled), fixed_img.affine
    )
    if debug:
        print("[registration] done")
    return resampled_img


def align_volumes(source_lf, target_hf, do_registration=False, debug=False):
    if debug:
        print("[align] loading and reorienting")
    lf_img = _load_canonical(source_lf)
    hf_img = _load_canonical(target_hf)
    if debug:
        print(f"[align] lf shape: {lf_img.shape}, hf shape: {hf_img.shape}")
        print("[align] resampling low-field to high-field grid")
    lf_resampled = _resample_to_reference(lf_img, hf_img, order=1)
    if do_registration:
        lf_resampled = _rigid_register_sitk(lf_resampled, hf_img, debug=debug)
    return lf_resampled.get_fdata(dtype=np.float32), hf_img.get_fdata(dtype=np.float32)


def registration(source_lf, target_hf, debug=False):
    """
    Register the low-field volume to the high-field volume using affine transformation.

    Args:
        source_lf: Path to low-field .nii.gz file
        target_hf: Path to high-field .nii.gz file """
    
    lf_data, hf_data = align_volumes(
        source_lf, target_hf, do_registration=True, debug=debug
    )
    return lf_data, hf_data


def visualize_alignment(lf_data, hf_data, slice_index=None, axis=2):
    if slice_index is None:
        slice_index = lf_data.shape[axis] // 2

    if axis == 0:
        lf_slice = lf_data[slice_index, :, :]
        hf_slice = hf_data[slice_index, :, :]
    elif axis == 1:
        lf_slice = lf_data[:, slice_index, :]
        hf_slice = hf_data[:, slice_index, :]
    else:
        lf_slice = lf_data[:, :, slice_index]
        hf_slice = hf_data[:, :, slice_index]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title("Low Field (Aligned)")
    axes[0].imshow(lf_slice, cmap="gray")
    axes[0].axis("off")

    axes[1].set_title("High Field (Reference)")
    axes[1].imshow(hf_slice, cmap="gray")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_blended(blended, slice_index=None, axis=2):
    if slice_index is None:
        slice_index = blended.shape[axis] // 2

    if axis == 0:
        blend_slice = blended[slice_index, :, :]
    elif axis == 1:
        blend_slice = blended[:, slice_index, :]
    else:
        blend_slice = blended[:, :, slice_index]

    plt.figure(figsize=(5, 5))
    plt.title("Blended (0.5 * LF + 0.5 * HF)")
    plt.imshow(blend_slice, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def pre_process_volume(source_lf, target_hf, do_registration=True, debug=False):
    if do_registration:
        lf_data, hf_data = registration(source_lf, target_hf, debug=debug)
    else:
        lf_data, hf_data = align_volumes(
            source_lf, target_hf, do_registration=False, debug=debug
        )
    #blended = 0.5 * lf_data + 0.5 * hf_data
    #visualize_alignment(lf_data, hf_data)
    #visualize_blended(blended)
    lf_normalized = normalize_volume(lf_data)
    hf_normalized = normalize_volume(hf_data)
    lf_patches = patch_volume(lf_normalized)
    hf_patches = patch_volume(hf_normalized)
    return lf_patches, hf_patches

def compute_mean_std(patches):
    mean = np.mean(patches)
    std = np.std(patches)
    return mean, std

def compute_difference(patches_lf, patches_hf):
    differences = patches_hf - patches_lf
    return differences

if __name__ == "__main__":
    source_lf = "mri_resolution/train/low_field/sample_001_lowfield.nii"
    target_hf = "mri_resolution/train/high_field/sample_001_highfield.nii"
    lf_patches, hf_patches = pre_process_volume(
        source_lf, target_hf, do_registration=False, debug=True
    )