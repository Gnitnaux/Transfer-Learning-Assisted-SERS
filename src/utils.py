"""
Utility functions for SERS data processing and analysis
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_spectra_train(directory):
    """
    Read and preprocess SERS spectral data from the specified directory for training.
    Args:
        directory (str): Path to the directory containing spectral data folders.
    Returns:
        tuple: (Raman_Shift, Intensity, Category, Concentration)
    """

    # read data from directory
    data_dict = {}
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            spectra_data = []
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.endswith('.csv'):
                    data = pd.read_csv(file_path, sep=',', skiprows=[0], names=['Raman Shift', 'Intensity'], encoding='GBK')
                    data_cut = data[(data['Raman Shift'] >= 400) & (data['Raman Shift'] <= 2000)]
                    spectra_data.append(data_cut)
            data_dict[folder] = spectra_data

    # data reshape
    # train data name: [DA/E/NE]_[concentration]uM_[replicate]
    Intensity_list = []
    Category_list = []
    Concentration_list = []
    Raman_Shift = None

    for folder, spectra in data_dict.items():
        for sp in spectra:
            if Raman_Shift is None:
                Raman_Shift = sp['Raman Shift'].values
            Intensity_list.append(sp['Intensity'].values)
            Category_list.append(folder.split('_')[0])
            Concentration_list.append((folder.split('_')[1]).split('u')[0]) 

    Intensity = np.array(Intensity_list)
    Category = np.array(Category_list)
    Concentration = np.array(Concentration_list, dtype=float)

    return Raman_Shift, Intensity, Category, Concentration

def read_spectra_predict(directory):
    """
    Read and preprocess SERS spectral data from the specified directory for prediction.
    Args:
        directory (str): Path to the directory containing spectral data folders.
    Returns:
        tuple: (Raman_Shift, Intensity, Concentrations)
    """

    # read data from directory
    data_dict = {}
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            spectra_data = []
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.endswith('.csv'):
                    data = pd.read_csv(file_path, sep=',', skiprows=[0], names=['Raman Shift', 'Intensity'], encoding='GBK')
                    data_cut = data[(data['Raman Shift'] >= 400) & (data['Raman Shift'] <= 2000)]
                    spectra_data.append(data_cut)
            data_dict[folder] = spectra_data

    # data reshape
    # predict data name: [DA]uM_[E]uM_[NE]uM_[replicate]
    Intensity_list = []
    Concentration_list = []
    Raman_Shift = None

    for folder, spectra in data_dict.items():
        for sp in spectra:
            if Raman_Shift is None:
                Raman_Shift = sp['Raman Shift'].values
            Intensity_list.append(sp['Intensity'].values)
            DA_con = (folder.split('_')[0]).split('u')[0]
            E_con = (folder.split('_')[1]).split('u')[0]
            NE_con = (folder.split('_')[2]).split('u')[0]
            Concentration_list.append([float(DA_con), float(E_con), float(NE_con)]) 

    Intensity = np.array(Intensity_list)
    Concentrations = np.array(Concentration_list, dtype=float)

    return Raman_Shift, Intensity, Concentrations


def spectra_normalization(Raman_Shift, Intensity, peak_position = 1480, peak_range = 20, plot = False):
    """
    Normalize SERS spectra based on a specific peak intensity.
    
    Args:
        Raman_Shift (np.ndarray): Array of Raman shift values.
        Intensity (np.ndarray): 2D array of intensity values (samples x features).
        peak_position (int): The Raman shift position of the peak to normalize against.
        peak_range (int): The range around the peak position to consider for normalization.
    
    Returns:
        np.ndarray: Normalized intensity array.
    """
    peak_indices = np.where((Raman_Shift >= peak_position - peak_range) & (Raman_Shift <= peak_position + peak_range))[0]
       
    normalized_Intensity = Intensity.copy()
    
    for i in range(Intensity.shape[0]):
        peak_intensity = np.max(Intensity[i, peak_indices])
        if peak_intensity != 0:
            normalized_Intensity[i, :] = Intensity[i, :] / peak_intensity
        else:
            normalized_Intensity[i, :] = Intensity[i, :]

    # min-max scaling to [0, 1]
    min_vals = np.min(normalized_Intensity, axis=1, keepdims=True)
    max_vals = np.max(normalized_Intensity, axis=1, keepdims=True)
    normalized_Intensity = (normalized_Intensity - min_vals) / (max_vals - min_vals + 1e-8)
    
    if plot:
        plt.figure(figsize=(10, 6))
        for i in range(Intensity.shape[0]):
            plt.plot(Raman_Shift, normalized_Intensity[i, :], label=f'Spectrum {i+1}')
        plt.xlabel('Raman Shift (cm⁻¹)')
        plt.ylabel('Normalized Intensity')
        plt.title('Normalized SERS Spectra')
        plt.show()
        
    return normalized_Intensity