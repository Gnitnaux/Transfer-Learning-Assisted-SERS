"""
Utility functions for SERS data processing and analysis
"""
import os
import pandas as pd
import numpy as np

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
                    spectra_data.append(data)
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
            Category_list.append(folder.split('_')[0])  # 只取文件夹名前部分作为标签
            Concentration_list.append((folder.split('_')[1]).split('u')[0])  # 浓度部分 

    Intensity = np.array(Intensity_list)        # shape = (光谱条数, 数据点数)
    Category = np.array(Category_list)        # 标签
    Concentration = np.array(Concentration_list, dtype=float)

    return Raman_Shift, Intensity, Category, Concentration

