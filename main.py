# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import os
import shutil
import errno
import numpy as np
import scipy.ndimage
from sklearn.preprocessing import StandardScaler
import nibabel as nib


def mkdir_p(path):
    print("mkdir_p")

    try:
        os.makedirs(path, mode=0o770)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

    return True


def resample_data(input_resample_data_array, data_array):
    print("resample_data")

    data_array_shape = data_array.shape

    resample_data_array = input_resample_data_array.copy()
    resample_data_array_shape = resample_data_array.shape

    if data_array_shape != resample_data_array_shape:
        zoom_factor = np.array(data_array_shape) / np.array(resample_data_array_shape)

        resample_data_array = scipy.ndimage.zoom(resample_data_array, zoom_factor, order=1, mode="nearest")

        input_resample_data_array = resample_data_array.copy()

    return input_resample_data_array


def preprocessing(data_array):
    print("preprocessing")

    standard_scaler_list = []

    if data_array.ndim > 3:
        number_of_time_points = data_array.shape[3]
    else:
        number_of_time_points = 1

    for i in range(number_of_time_points):
        if data_array.ndim > 3:
            current_data_array = data_array[:, :, :, i]
        else:
            current_data_array = data_array.copy()

        current_data_array_shape = current_data_array.shape
        current_data_array = current_data_array.reshape(-1, 1)

        standard_scaler_list.append(StandardScaler())
        current_data_array = standard_scaler_list[-1].fit_transform(current_data_array)

        if data_array.ndim > 3:
            data_array[:, :, :, i] = current_data_array.reshape(current_data_array_shape)
        else:
            data_array = current_data_array.reshape(current_data_array_shape)

    return data_array, standard_scaler_list


def affine_register(output_path, nifty_aladin_path, floating_data_path, reference_data_path):
    print("affine_register")

    res_output_path = "{0}/res.nii.gz".format(output_path)

    nifty_aladin_command = "{0} -ref {1} -flo {2} -res {3}".format(nifty_aladin_path, reference_data_path, floating_data_path, res_output_path)

    print(nifty_aladin_command)
    os.system(nifty_aladin_command)

    return res_output_path, reference_data_path


def register(output_path, nifty_reg_path, floating_data_path, reference_data_path):
    print("register")

    cpp_output_path = "{0}/cpp.nii.gz".format(output_path)
    res_output_path = "{0}/res.nii.gz".format(output_path)

    spacing = -5
    be = 1e-03
    le = 1e-02
    jl = 0.0
    ln = 3

    nifty_reg_command = "{0} -ref {1} -flo {2} -cpp {3} -res {4} -sx {5} -be {6} -le {7} -jl {8} -ln {9} -vel".format(nifty_reg_path, reference_data_path, floating_data_path, cpp_output_path, res_output_path, str(spacing), str(be), str(le), str(jl), str(ln))

    print(nifty_reg_command)
    os.system(nifty_reg_command)

    return res_output_path, reference_data_path


def inverse_preprocessing(data_array, standard_scaler_list):
    print("inverse_preprocessing")

    if data_array.ndim > 3:
        number_of_time_points = data_array.shape[3]
    else:
        number_of_time_points = 1

    for i in range(number_of_time_points):
        if data_array.ndim > 3:
            current_data_array = data_array[:, :, :, i]
        else:
            current_data_array = data_array.copy()

        current_data_array_shape = current_data_array.shape
        current_data_array = current_data_array.reshape(-1, 1)

        current_data_array = standard_scaler_list[i].inverse_transform(current_data_array)

        if data_array.ndim > 3:
            data_array[:, :, :, i] = current_data_array.reshape(current_data_array_shape)
        else:
            data_array = current_data_array.reshape(current_data_array_shape)

    return data_array


def main():
    print("main")

    output_path = "{0}/output/".format(os.getcwd())

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    mkdir_p(output_path)

    registration_output_path = "{0}/registration/".format(output_path)

    mkdir_p(registration_output_path)

    nifty_aladin_path = "/home/alex/Documents/niftyreg_install/bin/reg_aladin"
    nifty_reg_path = "/home/alex/Documents/niftyreg_install/bin/reg_f3d"

    floating_data_path = "/home/alex/Downloads/Dicom/SPECT_TT10_wSeg.nii.gz"
    reference_data_path = "/home/alex/Downloads/Dicom/MRI_T2_1av_wSeg.nii.gz"

    # floating_data_path = "/home/alex/Documents/jrmomo/interfile_to_pet/src/MLACF/gating/gated_image_data/dynamic/gated_image_data_1.nii.gz"
    # reference_data_path = "/home/alex/Documents/jrmomo/interfile_to_pet/src/MLACF/gating/gated_image_data/dynamic/gated_image_data_3.nii.gz"

    floating_data = nib.load(floating_data_path)
    reference_data = nib.load(reference_data_path)

    floating_data_array = np.nan_to_num(floating_data.get_fdata()[:, :, :, 0])
    reference_data_array = np.nan_to_num(reference_data.get_fdata()[:, :, :, 0])

    # floating_data_array = np.nan_to_num(floating_data.get_fdata())
    # reference_data_array = np.nan_to_num(reference_data.get_fdata())

    input_floating_data_array = floating_data_array.copy()
    input_reference_data_array = reference_data_array.copy()

    if floating_data_array.shape > reference_data_array.shape:
        reference_data_array = resample_data(reference_data_array, floating_data_array)

        preprocessed_data_voxel_sizes = floating_data.header.get_zooms()
    else:
        floating_data_array = resample_data(floating_data_array, reference_data_array)

        preprocessed_data_voxel_sizes = reference_data.header.get_zooms()

    floating_data_array, floating_data_array_standard_scaler = preprocessing(floating_data_array)
    reference_data_array, reference_data_array_standard_scaler = preprocessing(reference_data_array)

    preprocessed_floating_data = nib.Nifti1Image(floating_data_array,
                                                 np.array([[-preprocessed_data_voxel_sizes[0], 0.0, 0.0, (preprocessed_data_voxel_sizes[0] * (floating_data_array.shape[0] / 2.0)) - (preprocessed_data_voxel_sizes[0] / 2.0)],
                                                           [0.0, preprocessed_data_voxel_sizes[1], 0.0, - ((preprocessed_data_voxel_sizes[1] * (floating_data_array.shape[1] / 2.0)) - (preprocessed_data_voxel_sizes[1] / 2.0))],
                                                           [0.0, 0.0, preprocessed_data_voxel_sizes[2], - (preprocessed_data_voxel_sizes[3] / 2.0)],
                                                           [0.0, 0.0, 0.0, 1.0]]),
                                                 nib.Nifti1Header())
    preprocessed_reference_data = nib.Nifti1Image(reference_data_array,
                                                  np.array([[-preprocessed_data_voxel_sizes[0], 0.0, 0.0, (preprocessed_data_voxel_sizes[0] * (floating_data_array.shape[0] / 2.0)) - (preprocessed_data_voxel_sizes[0] / 2.0)],
                                                            [0.0, preprocessed_data_voxel_sizes[1], 0.0, - ((preprocessed_data_voxel_sizes[1] * (floating_data_array.shape[1] / 2.0)) - (preprocessed_data_voxel_sizes[1] / 2.0))],
                                                            [0.0, 0.0, preprocessed_data_voxel_sizes[2], - (preprocessed_data_voxel_sizes[3] / 2.0)],
                                                            [0.0, 0.0, 0.0, 1.0]]),
                                                  nib.Nifti1Header())

    preprocessed_floating_data_path = "{0}/preprocessed_floating_data.nii.gz".format(output_path)
    preprocessed_reference_data_path = "{0}/preprocessed_reference_data.nii.gz".format(output_path)

    nib.save(preprocessed_floating_data, preprocessed_floating_data_path)
    nib.save(preprocessed_reference_data, preprocessed_reference_data_path)

    # registered_floating_data_path, registered_reference_data_path = affine_register(registration_output_path,
    #                                                                                 nifty_aladin_path,
    #                                                                                 preprocessed_floating_data_path,
    #                                                                                 preprocessed_reference_data_path)
    registered_floating_data_path, registered_reference_data_path = register(registration_output_path, nifty_reg_path,
                                                                             preprocessed_floating_data_path,
                                                                             preprocessed_reference_data_path)

    registered_floating_data = nib.load(registered_floating_data_path)
    registered_reference_data = nib.load(registered_reference_data_path)

    floating_data_array = np.nan_to_num(registered_floating_data.get_fdata())
    reference_data_array = np.nan_to_num(registered_reference_data.get_fdata())

    floating_data_array = inverse_preprocessing(floating_data_array, floating_data_array_standard_scaler)
    reference_data_array = inverse_preprocessing(reference_data_array, reference_data_array_standard_scaler)

    # floating_data_array = resample_data(floating_data_array, input_floating_data_array)
    # reference_data_array = resample_data(reference_data_array, input_reference_data_array)

    floating_data_voxel_size = floating_data.header.get_zooms()
    reference_data_voxel_size = reference_data.header.get_zooms()

    output_floating_data = nib.Nifti1Image(floating_data_array,
                                           np.array([[-preprocessed_data_voxel_sizes[0], 0.0, 0.0, (preprocessed_data_voxel_sizes[0] * (floating_data_array.shape[0] / 2.0)) - (preprocessed_data_voxel_sizes[0] / 2.0)],
                                                     [0.0, preprocessed_data_voxel_sizes[1], 0.0, - ((preprocessed_data_voxel_sizes[1] * (floating_data_array.shape[1] / 2.0)) - (preprocessed_data_voxel_sizes[1] / 2.0))],
                                                     [0.0, 0.0, preprocessed_data_voxel_sizes[2], - (preprocessed_data_voxel_sizes[3] / 2.0)],
                                                     [0.0, 0.0, 0.0, 1.0]]),
                                           nib.Nifti1Header())
    output_reference_data = nib.Nifti1Image(reference_data_array,
                                            np.array([[-preprocessed_data_voxel_sizes[0], 0.0, 0.0, (preprocessed_data_voxel_sizes[0] * (floating_data_array.shape[0] / 2.0)) - (preprocessed_data_voxel_sizes[0] / 2.0)],
                                                      [0.0, preprocessed_data_voxel_sizes[1], 0.0, - ((preprocessed_data_voxel_sizes[1] * (floating_data_array.shape[1] / 2.0)) - (preprocessed_data_voxel_sizes[1] / 2.0))],
                                                      [0.0, 0.0, preprocessed_data_voxel_sizes[2], - (preprocessed_data_voxel_sizes[3] / 2.0)],
                                                      [0.0, 0.0, 0.0, 1.0]]),
                                            nib.Nifti1Header())

    nib.save(output_floating_data, "{0}/output_floating_data.nii.gz".format(output_path))
    nib.save(output_reference_data, "{0}/output_reference_data.nii.gz".format(output_path))


if __name__ == "__main__":
    main()
