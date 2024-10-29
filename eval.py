#!/usr/bin/env python

"""
eval.py

This module provides functions to compute various error metrics between 3D predicted and reference 
data arrays. The primary function, `all_metrics()`, returns a dictionary of all computed metrics.

Metrics included: RMSE, NRMSE, HFEN, XSIM, MAD, CC, NMI, GXE.

Example:
    >>> import numpy as np
    >>> import metrics
    >>> pred_data = np.random.rand(100, 100, 100)
    >>> ref_data = np.random.rand(100, 100, 100)
    >>> roi = np.random.randint(0, 2, size=(100, 100, 100), dtype=bool)
    >>> metrics = metrics.all_metrics(pred_data, ref_data, roi)

Authors: Boyi Du <boyi.du@uq.net.au>, Ashley Stewart <ashley.stewart@uq.edu.au>

"""

import json
import argparse
import os
import numpy as np
import csv
import nibabel as nib

from sklearn.metrics import root_mean_squared_error
from skimage.metrics import structural_similarity
from skimage.metrics import normalized_mutual_information
from skimage.measure import pearson_corr_coeff
from skimage.measure import shannon_entropy
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import gaussian_laplace

def calculate_gradient_magnitude(pred_data):
    """
    Calculate the gradient magnitude of the input data, with padding to handle small arrays.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.

    Returns
    -------
    tuple
        The mean and standard deviation of the gradient magnitude.
    """
    # Minimum size required to calculate the gradient
    min_size = 2
    
    # Function to pad the array if any dimension is smaller than the required size
    def pad_if_necessary(array):
        padding = [(0, max(min_size - dim_size, 0)) for dim_size in array.shape]
        return np.pad(array, padding, mode='constant', constant_values=0)

    # Pad the predicted data if necessary
    pred_data_padded = pad_if_necessary(pred_data)

    # Amplify the data to make gradient computation more sensitive
    amplified_data = pred_data_padded * 10
    
    # Calculate the gradients
    gradients = np.gradient(amplified_data)
    
    # Calculate the gradient magnitude
    grad_magnitude = np.sqrt(sum([g**2 for g in gradients]))
    
    # Return the mean and standard deviation of the gradient magnitude, ignoring zero regions
    return grad_magnitude[pred_data_padded != 0].mean(), grad_magnitude[pred_data_padded != 0].std()

def calculate_total_variation(pred_data, weight=0.1):
    amplified_data = pred_data * 1000
    denoised_image = denoise_tv_chambolle(amplified_data, weight=weight)
    nib.save(nib.Nifti1Image(dataobj=denoised_image, affine=None, header=None), "OUT.nii")
    tv_norm = np.sum(np.abs(amplified_data[pred_data != 0] - denoised_image[pred_data != 0]))  # Total variation norm
    tv_normalized = tv_norm / np.size(pred_data[pred_data != 0])  # Normalize by total number of non-zero voxels
    return tv_normalized

def calculate_entropy(pred_data):
    entropy_value = shannon_entropy(pred_data)
    max_entropy = np.log2(np.prod(pred_data.shape))  # Maximum possible entropy
    return entropy_value / max_entropy  # Normalised entropy

def calculate_cnr(region1, region2):
    mean_diff = np.abs(np.mean(region1) - np.mean(region2))
    noise = np.std(region1) + np.std(region2)
    if noise == 0:
        return float('inf')  # Handle the case where noise is zero
    return mean_diff / noise

def calculate_snr(pred_data, roi_foreground, roi_background):
    signal = np.mean(pred_data[roi_foreground == 1])  # Mean intensity in the signal region
    noise = np.std(pred_data[roi_background == 1])  # Standard deviation in the background
    if noise == 0:
        return float('inf')  # Handle the case where noise is zero
    return signal / noise

def calculate_edge_strength(pred_data):
    amplified_data = pred_data * 100.0
    edges = gaussian_laplace(amplified_data, sigma=1.5)
    non_zero_edges = edges[pred_data != 0]
    return np.var(non_zero_edges)


def calculate_rmse(pred_data, ref_data):
    """
    Calculate the Root Mean Square Error (RMSE) between the predicted and reference data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.

    Returns
    -------
    float
        The calculated RMSE value.

    """
    return root_mean_squared_error(pred_data, ref_data)

def calculate_nrmse(pred_data, ref_data):
    """
    Calculate the Normalized Root Mean Square Error (NRMSE) between the predicted and reference data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.

    Returns
    -------
    float
        The calculated NRMSE value.

    References
    ----------
    .. [1] https://github.com/scikit-image/scikit-image/blob/v0.21.0/skimage/metrics/simple_metrics.py#L50-L108
    """
    rmse = calculate_rmse(pred_data, ref_data)
    nrmse = rmse * np.sqrt(len(ref_data)) / np.linalg.norm(ref_data) * 100 # Frobenius norm
    return nrmse

def calculate_hfen(pred_data, ref_data):
    """
    Calculate the High Frequency Error Norm (HFEN) between the predicted and reference data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.

    Returns
    -------
    float
        The calculated HFEN value.
    References
    ----------
    .. [1] https://doi.org/10.1002/mrm.26830

    """
    LoG_pred = gaussian_laplace(pred_data, sigma = 1.5)
    LoG_ref = gaussian_laplace(ref_data, sigma = 1.5)
    hfen = np.linalg.norm(LoG_ref - LoG_pred)/np.linalg.norm(LoG_ref)
    return hfen

def calculate_xsim(pred_data, ref_data, data_range=None):
    """
    Calculate the structural similarity (XSIM) between the predicted and reference data.
    Pads the arrays with zeros if necessary to avoid errors during the SSIM calculation.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.
    data_range : float
        Expected data range.

    Returns
    -------
    float
        The calculated structural similarity value.

    References
    ----------
    .. [1] Milovic, C., et al. (2024). XSIM: A structural similarity index measure optimized for MRI QSM. Magnetic Resonance in Medicine. doi:10.1002/mrm.30271
    """
    # Determine the minimum size for the SSIM window
    min_size = 7

    # Function to pad arrays if any dimension is smaller than min_size
    def pad_if_necessary(array):
        padding = [(0, max(min_size - dim_size, 0)) for dim_size in array.shape]
        return np.pad(array, padding, mode='constant', constant_values=0)

    # Pad pred_data and ref_data if necessary
    pred_data_padded = pad_if_necessary(pred_data)
    ref_data_padded = pad_if_necessary(ref_data)

    # Determine the appropriate win_size
    win_size = min(min(pred_data_padded.shape), min_size)

    # Set data range if not provided
    if data_range is None:
        data_range = ref_data_padded.max() - ref_data_padded.min()

    # Calculate the structural similarity index (XSIM)
    xsim = structural_similarity(pred_data_padded, ref_data_padded, win_size=win_size, K1=0.01, K2=0.001, data_range=data_range)
    
    return xsim

def calculate_mad(pred_data, ref_data):
    """
    Calculate the Mean Absolute Difference (MAD) between the predicted and reference data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.

    Returns
    -------
    float
        The calculated MAD value.

    """
    mad = np.mean(np.abs(pred_data - ref_data))
    return mad

def calculate_gxe(pred_data, ref_data):
    """
    Calculate the gradient difference error (GXE) between the predicted and reference data.
    Pads the arrays with zeros if necessary to avoid errors during gradient calculation.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.

    Returns
    -------
    float
        The calculated GXE value.

    """
    # Determine the minimum size for gradient calculation (at least 2 elements in each dimension)
    min_size = 2
    
    # Function to pad arrays if any dimension is smaller than min_size
    def pad_if_necessary(array):
        padding = [(0, max(min_size - dim_size, 0)) for dim_size in array.shape]
        return np.pad(array, padding, mode='constant', constant_values=0)
    
    # Pad pred_data and ref_data if necessary
    pred_data_padded = pad_if_necessary(pred_data)
    ref_data_padded = pad_if_necessary(ref_data)
    
    # Compute the gradient difference error
    gxe = np.sqrt(np.mean((np.array(np.gradient(pred_data_padded)) - np.array(np.gradient(ref_data_padded))) ** 2))
    
    return gxe


def get_bounding_box(roi):
    """
    Calculate the bounding box of a 3D region of interest (ROI).

    Parameters
    ----------
    roi : numpy.ndarray
        A 3D numpy array representing a binary mask of the ROI,
        where 1 indicates an object of interest and 0 elsewhere.

    Returns
    -------
    bbox : tuple
        A tuple of slice objects representing the bounding box of the ROI. This can be 
        directly used to slice numpy arrays.

    Example
    -------
    >>> mask = np.random.randint(0, 2, size=(100, 100, 100))
    >>> bbox = get_bounding_box(mask)
    >>> sliced_data = data[bbox]

    Notes
    -----
    The function works by identifying the min and max coordinates of the ROI along 
    each axis. These values are used to generate a tuple of slice objects.
    The function will work for ROIs of arbitrary dimension, not just 3D.
    """
    coords = np.array(roi.nonzero())
    min_coords = coords.min(axis=1)
    max_coords = coords.max(axis=1) + 1
    return tuple(slice(min_coords[d], max_coords[d]) for d in range(roi.ndim))

def all_metrics(pred_data, ref_data=None, roi=None, segmentation=None, labels=None):
    """
    Calculate various error and quality metrics between the predicted data and the reference data (optional).

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray, optional
        Reference data as a numpy array. If not provided, only quality metrics without a reference are computed.
    roi : numpy.ndarray, optional
        A binary mask defining a region of interest within the data. If not provided, the full extent of pred_data is used.
    segmentation : numpy.ndarray, optional
        A segmentation mask where each distinct integer label corresponds to a different ROI.
    labels : dict, optional
        A dictionary mapping segmentation values (integers) to human-readable names.

    Returns
    -------
    dict
        A dictionary of calculated metrics, including RMSE, NRMSE, HFEN, XSIM, MAD, CC, NMI, GXE, and quality measures
        such as gradient magnitude, total variation, entropy, CNR, SNR, and edge strength.
    """
    
    # Define the region of interest if not provided
    metrics_by_roi = {}

    # If segmentation is provided, calculate metrics for each ROI label
    if segmentation is not None:
        unique_labels = np.unique(segmentation)
        unique_labels = unique_labels[unique_labels != 0]  # Ignore label 0 if it's background

        for label in unique_labels:
            # Use provided label names if available, otherwise fallback to "ROI_{label}"
            label_name = labels.get(label, f"ROI {label}") if labels else f"ROI {label}"
            
            print(f"[INFO] Computing metrics for {label_name}...")
            seg_roi = segmentation == label

            metrics_by_roi[label_name] = calculate_metrics_for_roi(pred_data, ref_data, seg_roi)

    print(f"[INFO] Computing metrics for all ROIs...")
    metrics_by_roi["All"] = calculate_metrics_for_roi(pred_data, ref_data)

    return metrics_by_roi


def calculate_metrics_for_roi(pred_data, ref_data=None, roi=None):
    """
    Helper function to compute metrics for a specific ROI.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray, optional
        Reference data as a numpy array.
    roi : numpy.ndarray
        A binary mask defining the region of interest (ROI) within the data.

    Returns
    -------
    dict
        A dictionary of metrics for the given ROI.
    """

    if roi is None:
        roi = np.array(pred_data != 0, dtype=bool)
    bbox = get_bounding_box(roi)
    roi = np.array(roi[bbox], dtype=bool)
    pred_data = pred_data[bbox] * roi
    if ref_data is not None:
        ref_data = ref_data[bbox] * roi
    
    d = {}

    if ref_data is not None:
        d['RMSE'] = calculate_rmse(pred_data[roi], ref_data[roi])
        d['NRMSE'] = calculate_nrmse(pred_data[roi], ref_data[roi])
        d['HFEN'] = calculate_hfen(pred_data, ref_data)
        d['MAD'] = calculate_mad(pred_data[roi], ref_data[roi])
        d['GXE'] = calculate_gxe(pred_data, ref_data)
        d['XSIM'] = calculate_xsim(pred_data, ref_data)
        d['CC'] = pearson_corr_coeff(pred_data[roi], ref_data[roi])
        d['NMI'] = normalized_mutual_information(pred_data[roi], ref_data[roi])

    d['Minimum'] = pred_data[roi].min()
    d['Maximum'] = pred_data[roi].max()
    d['Mean'] = pred_data[roi].mean()
    d['Median'] = np.median(pred_data[roi])
    d['Standard deviation'] = pred_data[roi].std()
    d['Gradient Mean'], d['Gradient Std'] = calculate_gradient_magnitude(pred_data)
    d['Total Variation'] = calculate_total_variation(pred_data)
    d['Entropy'] = calculate_entropy(pred_data)
    d['Edge Strength'] = calculate_edge_strength(pred_data)

    return d

def save_as_csv(metrics_dict, filepath):
    """
    Save the metrics as a CSV file

    Parameters
    ----------
    metrics_dict : dict
        A dictionary containing the metrics for each ROI.
    filepath : str
        The path to the file to save the results.
    """
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Region", "Value"])
        
        # Iterate over each ROI in the metrics dictionary
        for roi_label, metrics in metrics_dict.items():
            for key, value in metrics.items():
                writer.writerow([key, roi_label, value])


def save_as_markdown(metrics_dict, filepath):
    """
    Save the metrics as a markdown table

    Parameters
    ----------
    metrics_dict : dict
        A dictionary containing the metrics for each ROI.
    filepath : str
        The path to the file to save the results.
    """
    # Find the longest metric name, region, and value across all ROIs
    max_metric_len = max(len(str(key)) for roi_label, metrics in metrics_dict.items() for key in metrics.keys())
    max_region_len = max(len(str(roi_label)) for roi_label in metrics_dict.keys())
    max_value_len = max(len(str(f"{value[0]:.6f}")) if isinstance(value, tuple) and len(value) == 2 else len(f"{value:.6f}") 
                        for metrics in metrics_dict.values() for value in metrics.values())

    # Create the table with dynamic widths
    with open(filepath, 'w') as file:
        file.write(f"| {'Metric'.ljust(max_metric_len)} | {'Region'.ljust(max_region_len)} | {'Value'.ljust(max_value_len)} |\n")
        file.write(f"|{'-' * (max_metric_len + 2)}|{'-' * (max_region_len + 2)}|{'-' * (max_value_len + 2)}|\n")
        
        for roi_label, metrics in metrics_dict.items():
            for key, value in metrics.items():
                if isinstance(value, tuple) and len(value) == 2:  # Assuming it's the PearsonRResult
                    file.write(f"| {key.ljust(max_metric_len)} | {roi_label.ljust(max_region_len)} | {value[0]:.6f} |\n")
                    file.write(f"| {' '.ljust(max_metric_len)} | {' '.ljust(max_region_len)} | {value[1]:.6f} |\n")
                else:
                    file.write(f"| {key.ljust(max_metric_len)} | {roi_label.ljust(max_region_len)} | {value:.6f} |\n")


def save_as_json(metrics_dict, filepath):
    """
    Save the metrics as a JSON file

    Parameters
    ----------
    metrics_dict : dict
        A dictionary containing the metrics.
    filepath : str
        The path to the file to save the results.
    """
    json_data = []
    
    # Iterate over each ROI in the metrics dictionary and split Metric/Region
    for roi_label, metrics in metrics_dict.items():
        for key, value in metrics.items():
            json_data.append({
                "Metric": key,
                "Region": roi_label,
                "Value": value
            })

    # Save as JSON
    with open(filepath, 'w') as file:
        json.dump(json_data, file, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Compute metrics for 3D images.')
    parser.add_argument('--ground_truth', type=str, help='Path to the ground truth NIFTI image (optional).')
    parser.add_argument('--estimate', type=str, required=True, help='Path to the reconstructed NIFTI image.')
    parser.add_argument('--roi', type=str, help='Path to the ROI NIFTI image (optional).')
    parser.add_argument('--segmentation', type=str, help='Path to the segmentation NIFTI image (optional, if provided will compute metrics for each ROI).')
    parser.add_argument('--output_dir', type=str, default='./', help='Directory to save metrics.')
    args = parser.parse_args()

    # Load reconstructed image
    print("[INFO] Loading reconstructed image...")
    recon_img = nib.load(args.estimate).get_fdata()

    # Load ground truth image (if provided)
    if args.ground_truth:
        print("[INFO] Loading ground truth image...")
        gt_img = nib.load(args.ground_truth).get_fdata()
    else:
        gt_img = None

    # Load ROI (if provided)
    if args.roi:
        print("[INFO] Loading ROI image...")
        roi_img = np.array(nib.load(args.roi).get_fdata(), dtype=bool)
    else:
        roi_img = None

    # Load segmentation (if provided)
    if args.segmentation:
        print("[INFO] Loading segmentation image...")
        segmentation_img = nib.load(args.segmentation).get_fdata().astype(int)
    else:
        segmentation_img = None

    # Compute metrics
    print("[INFO] Computing metrics...")
    metrics = all_metrics(recon_img, ref_data=gt_img, roi=roi_img, segmentation=segmentation_img)

    # Save metrics
    print(f"[INFO] Saving results to {args.output_dir}...")
    csv_path = os.path.join(args.output_dir, 'metrics.csv')
    md_path = os.path.join(args.output_dir, 'metrics.md')
    json_path = os.path.join(args.output_dir, 'metrics.json')

    save_as_csv(metrics, csv_path)
    save_as_markdown(metrics, md_path)
    save_as_json(metrics, json_path)

    print(f"[INFO] Metrics saved to {csv_path}, {md_path}, and {json_path}")


if __name__ == "__main__":
    main()

