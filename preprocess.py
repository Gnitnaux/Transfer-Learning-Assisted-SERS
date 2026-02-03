#!/usr/bin/env python3
"""
Preprocessing program for SERS data
This module handles data preprocessing operations including:
- SG (Savitzky-Golay) filtering for spectral smoothing
- AirPLS baseline correction
- Batch processing of train and test datasets
"""

import os
import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from scipy import signal
from scipy.sparse import diags, eye, csc_matrix
from scipy.sparse.linalg import spsolve


def SG(data, window_length, polyorder):
    """
    Apply Savitzky-Golay filter to smooth spectral data.
    
    Args:
        data: Input data array (can be 1D or 2D)
        window_length: Length of the filter window (must be odd)
        polyorder: Order of the polynomial used to fit the samples
        
    Returns:
        Filtered data array
    """
    return signal.savgol_filter(data, window_length, polyorder)


def WhittakerSmooth(x, w, lambda_, differences=1):
    """
    Whittaker smoother for baseline estimation.
    
    Args:
        x: Input signal
        w: Weights
        lambda_: Smoothing parameter
        differences: Order of differences
        
    Returns:
        Smoothed baseline
    """
    X = np.matrix(x)
    m = X.size
    E = eye(m, format='csc')
    for i in range(differences):
        E = E[1:] - E[:-1]
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * E.T * E))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)


def airPLS(x, lambda_=1e8, porder=3, itermax=15):
    """
    Adaptive Iteratively Reweighted Penalized Least Squares for baseline correction.
    
    Args:
        x: Input signal
        lambda_: Smoothing parameter (larger = smoother baseline)
        porder: Order of differences in penalty
        itermax: Maximum number of iterations
        
    Returns:
        Estimated baseline
    """
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
            if (i == itermax): 
                print('WARNING: max iteration reached!')
            break
        w[d >= 0] = 0
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    return z


def process_single_spectrum(data, window_length=7, polyorder=3, lambda_val=1e6, porder=3):
    """
    Process a single spectrum using SG filtering and AirPLS baseline correction.
    
    Args:
        data: DataFrame with columns [Raman Shift, Intensity]
        window_length: SG filter window length
        polyorder: SG filter polynomial order
        lambda_val: AirPLS lambda parameter
        porder: AirPLS polynomial order
        
    Returns:
        DataFrame with processed spectrum [Raman Shift, Processed Intensity]
    """
    x = data.iloc[:, 0].values  # Raman Shift (reference)
    y = data.iloc[:, 1].values  # Intensity
    
    # Step 1: Apply SG filtering
    # Stack current spectrum with reference for processing
    merge1 = np.row_stack((y, x))
    sg_result = SG(merge1, window_length, polyorder)
    sg_filtered = sg_result[0]  # Take the first row as SG filtered result
    
    # Step 2: Apply AirPLS baseline correction
    # Stack SG filtered result with reference
    merge2 = np.row_stack((sg_filtered, x))
    
    # Apply AirPLS to both rows
    data_AirPLS = merge2.copy()
    for j in range(merge2.shape[0]):
        data_AirPLS[j] = merge2[j] - airPLS(merge2[j], lambda_=lambda_val, porder=porder)
    
    final_result = data_AirPLS[0]  # Take the first row as final result
    
    # Return as DataFrame
    processed_data = pd.DataFrame({
        'Raman Shift': x, 
        'Processed Intensity': final_result
    })
    
    return processed_data


def process_folder(input_folder, output_folder, window_length=7, polyorder=3, 
                   lambda_val=1e6, porder=3, prefix="processed_"):
    """
    Process all CSV files in a folder and its subfolders.
    
    Args:
        input_folder: Path to input folder containing subfolders with CSV files
        output_folder: Path to output folder
        window_length: SG filter window length
        polyorder: SG filter polynomial order
        lambda_val: AirPLS lambda parameter
        porder: AirPLS polynomial order
        prefix: Prefix for output filenames
        
    Returns:
        Dictionary mapping subfolder names to processed data
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all subfolders
    subfolders = [f for f in input_path.iterdir() if f.is_dir()]
    
    if not subfolders:
        print(f"Warning: No subfolders found in {input_folder}")
        return {}
    
    print(f"\nProcessing {len(subfolders)} subfolders in {input_folder}...")
    
    all_processed = {}
    
    for subfolder in subfolders:
        subfolder_name = subfolder.name
        print(f"  Processing subfolder: {subfolder_name}")
        
        # Create output subfolder
        output_subfolder = output_path / subfolder_name
        output_subfolder.mkdir(parents=True, exist_ok=True)
        
        # Find all CSV files in this subfolder
        csv_files = list(subfolder.glob("*.csv"))
        
        if not csv_files:
            print(f"    Warning: No CSV files found in {subfolder_name}")
            continue
        
        print(f"    Found {len(csv_files)} CSV files")
        
        processed_files = {}
        all_spectra = []  # To calculate mean spectrum
        raman_shift = None
        
        for csv_file in csv_files:
            try:
                # Read CSV file
                # Skip first row if it contains headers, use GBK encoding as in original code
                data = pd.read_csv(csv_file, sep=',', skiprows=[0], 
                                  names=['Raman Shift', 'Intensity'], 
                                  encoding='GBK')
                
                # Store raman shift values (should be same for all files)
                if raman_shift is None:
                    raman_shift = data.iloc[:, 0].values
                
                # Process the spectrum
                processed_data = process_single_spectrum(
                    data, window_length, polyorder, lambda_val, porder
                )
                
                # Save individual processed file
                output_filename = f"{prefix}{csv_file.stem}.csv"
                output_filepath = output_subfolder / output_filename
                processed_data.to_csv(output_filepath, index=False)
                
                # Store for mean calculation
                processed_files[csv_file.name] = processed_data
                all_spectra.append(processed_data['Processed Intensity'].values)
                
            except Exception as e:
                print(f"    Error processing {csv_file.name}: {str(e)}")
                continue
        
        # Calculate and save mean spectrum for this subfolder
        if all_spectra:
            mean_spectrum = np.mean(all_spectra, axis=0)
            mean_data = pd.DataFrame({
                'Raman Shift': raman_shift,
                f'{subfolder_name} Mean': mean_spectrum
            })
            mean_filepath = output_subfolder / f"{prefix}{subfolder_name}_mean.csv"
            mean_data.to_csv(mean_filepath, index=False)
            print(f"    Saved mean spectrum to {mean_filepath.name}")
        
        all_processed[subfolder_name] = processed_files
        print(f"    Completed {subfolder_name}: {len(processed_files)} files processed")
    
    return all_processed


def run_preprocessing(data_dir, output_dir, window_length=7, polyorder=3, 
                     lambda_val=1e6, porder=3, prefix="processed_"):
    """
    Main preprocessing pipeline for both train and test datasets.
    
    Args:
        data_dir: Path to raw data directory (should contain train/ and test/ subfolders)
        output_dir: Path to output directory for preprocessed data
        window_length: SG filter window length
        polyorder: SG filter polynomial order
        lambda_val: AirPLS lambda parameter
        porder: AirPLS polynomial order
        prefix: Prefix for output filenames
    """
    print("\n" + "=" * 60)
    print("SERS Data Preprocessing Pipeline")
    print("=" * 60)
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nPreprocessing parameters:")
    print(f"  SG Filter - window_length: {window_length}, polyorder: {polyorder}")
    print(f"  AirPLS - lambda: {lambda_val}, porder: {porder}")
    print(f"  Output prefix: {prefix}")
    print("=" * 60)
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
        print(f"\nNote: Adjusted window_length to {window_length} (must be odd)")
    
    # Process train folder
    train_input = data_path / "train"
    train_output = output_path / "train"
    
    if train_input.exists():
        print("\n" + "-" * 60)
        print("Processing TRAIN dataset")
        print("-" * 60)
        train_processed = process_folder(
            train_input, train_output, window_length, polyorder, 
            lambda_val, porder, prefix
        )
        print(f"\nTrain processing complete: {len(train_processed)} categories processed")
    else:
        print(f"\nWarning: Train folder not found at {train_input}")
    
    # Process test folder
    test_input = data_path / "test"
    test_output = output_path / "test"
    
    if test_input.exists():
        print("\n" + "-" * 60)
        print("Processing TEST dataset")
        print("-" * 60)
        test_processed = process_folder(
            test_input, test_output, window_length, polyorder, 
            lambda_val, porder, prefix
        )
        print(f"\nTest processing complete: {len(test_processed)} categories processed")
    else:
        print(f"\nWarning: Test folder not found at {test_input}")
    
    print("\n" + "=" * 60)
    print("Preprocessing completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess SERS data for transfer learning using SG filtering and AirPLS baseline correction"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to raw data directory (should contain train/ and test/ subfolders)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/preprocessed",
        help="Path to output directory for preprocessed data"
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=7,
        help="SG filter window length (must be odd, default: 7)"
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        default=3,
        help="SG filter polynomial order (default: 3)"
    )
    parser.add_argument(
        "--lambda-val",
        type=float,
        default=1e6,
        help="AirPLS lambda parameter (default: 1e6)"
    )
    parser.add_argument(
        "--porder",
        type=int,
        default=3,
        help="AirPLS polynomial order (default: 3)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="processed_",
        help="Prefix for output filenames (default: 'processed_')"
    )
    
    args = parser.parse_args()
    
    run_preprocessing(
        args.data_dir, 
        args.output_dir,
        args.window_length,
        args.polyorder,
        args.lambda_val,
        args.porder,
        args.prefix
    )

