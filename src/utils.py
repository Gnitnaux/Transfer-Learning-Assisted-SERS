"""
Utility functions for SERS data processing and analysis
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DA_PROB_THRESHOLD = 0.5
E_PROB_THRESHOLD = 0.5
NE_PROB_THRESHOLD = 0.5

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
                    data_cut = data[(data['Raman Shift'] >= 330) & (data['Raman Shift'] <= 1600)]
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

def read_spectra_test(directory):
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
                    data_cut = data[(data['Raman Shift'] >= 330) & (data['Raman Shift'] <= 1600)]
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

def read_spectra_unknown(directory):
    """
    Read and preprocess unknown SERS spectral data from the specified directory.
    Args:
        directory (str): Path to the directory containing spectral data folders.
    Returns:
        tuple: (Raman_Shift, Intensity, Labels)
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
                    data_cut = data[(data['Raman Shift'] >= 330) & (data['Raman Shift'] <= 1600)]
                    spectra_data.append(data_cut)
            data_dict[folder] = spectra_data

    # data reshape
    # predict data name: [DA]uM_[E]uM_[NE]uM_[replicate]
    Intensity_list = []
    Label_list = []
    Raman_Shift = None

    for folder, spectra in data_dict.items():
        for sp in spectra:
            if Raman_Shift is None:
                Raman_Shift = sp['Raman Shift'].values
            Intensity_list.append(sp['Intensity'].values)
            Label_list.append(folder)

    Intensity = np.array(Intensity_list)
    Labels = np.array(Label_list, dtype=str)

    return Raman_Shift, Intensity, Labels

def spectra_normalization(Raman_Shift, Intensity, peak_position = 1480, peak_range = 20, plot = False, mode = 'train'):
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
        plt.savefig(f'visualization/Normalized_SERS_Spectra_{mode}.png', dpi = 600)
        plt.show(block = False)
        # wait for 5s then close the plot
        plt.pause(5)
        plt.close()
        
    return normalized_Intensity


def digital_mix_ID(Raman_Shift, Intensity, Category, CA, data_concentration, num_mix = 1000, Range = [0.5, 10]):
    """
    Create digital mixed spectra for identification model training.
    
    Args:
        Raman_Shift (np.ndarray): Array of Raman shift values.
        Intensity (np.ndarray): 2D array of intensity values (samples x features).
        Category (np.ndarray): Array of category labels for each sample.
        CA (str): The target chemical agent for identification.
        data_concentration (float): The concentration value for the target chemical agent.
        num_mix (int): The number of mixed spectra to generate.
        Range (list): The range of concentrations for the mixture components.
    Returns:
        Intensity_mix (np.ndarray): 2D array of digitally mixed intensity values.
        Label_mix (0/1): Array of binary labels indicating presence (1) or absence (0) of the target CA.
    """

    Index_DA = np.where(Category == 'DA')[0]
    Index_E = np.where(Category == 'E')[0]
    Index_NE = np.where(Category == 'NE')[0]
    Index_BA = np.where(Category == 'BA')[0]

    Intensity_DA = Intensity[Index_DA]
    Intensity_E = Intensity[Index_E]
    Intensity_NE = Intensity[Index_NE]
    Intensity_BA = Intensity[Index_BA]

    # Generate random concentration of mixture
    np.random.seed(42)  # For reproducibility
    mix_concentration = np.random.uniform(Range[0], Range[1], num_mix)
    ratio_CA = mix_concentration / data_concentration

    ratio_CAs = []
    Label_mix = []
    # Generate radom ratio containing CA
    for i in range(num_mix//2):
        ratio_target = (np.random.uniform(0.3, 1, 1)[0]) * ratio_CA[i]  # Ratio of CA in the mixture
        ratio_other_1 = (np.random.uniform(0, 1 - ratio_target, 1)[0]) * ratio_CA[i]  # Ratio of other component 1
        ratio_other_2 = (1 - ratio_target - ratio_other_1) * ratio_CA[i]  # Ratio of other component
        if CA == 'DA':
            ratio_CAs.append([ratio_target, ratio_other_1, ratio_other_2, 1- ratio_CA[i]])
        elif CA == 'E':
            ratio_CAs.append([ratio_other_1, ratio_target, ratio_other_2, 1- ratio_CA[i]])
        elif CA == 'NE':
            ratio_CAs.append([ratio_other_1, ratio_other_2, ratio_target, 1- ratio_CA[i]])
        Label_mix.append(1)  # Label 1 for spectra containing CA

    for i in range(num_mix - num_mix//2):
        ratio_other_1 = (np.random.uniform(0, 1, 1)[0]) * ratio_CA[i]  # Ratio of other component 1
        ratio_other_2 = (np.random.uniform(0, 1 - ratio_other_1, 1)[0]) * ratio_CA[i]  # Ratio of other component 2
        if CA == 'DA':
            ratio_CAs.append([0, ratio_other_1, ratio_other_2, 1- ratio_CA[i]])
        elif CA == 'E':
            ratio_CAs.append([ratio_other_1, 0, ratio_other_2, 1- ratio_CA[i]])
        elif CA == 'NE':
            ratio_CAs.append([ratio_other_1, ratio_other_2, 0, 1- ratio_CA[i]])
        Label_mix.append(0)  # Label 0 for spectra not containing CA

    # print(ratio_CAs[:10])

    # Linear combination of spectra based on concentration and ratio
    Intensity_mix = []
    for i in range(num_mix):
        selected_DA = Intensity_DA[np.random.choice(Intensity_DA.shape[0])]
        selected_E = Intensity_E[np.random.choice(Intensity_E.shape[0])]
        selected_NE = Intensity_NE[np.random.choice(Intensity_NE.shape[0])]
        selected_BA = Intensity_BA[np.random.choice(Intensity_BA.shape[0])]

        mixed_spectrum = (ratio_CAs[i][0] * selected_DA + ratio_CAs[i][1] * selected_E +
                          ratio_CAs[i][2] * selected_NE + ratio_CAs[i][3] * selected_BA)
        Intensity_mix.append(mixed_spectrum)

    Intensity_mix = np.array(Intensity_mix)
    Label_mix = np.array(Label_mix)

    # plt.figure(figsize=(10, 6))
    # for i in range(10):  # Plot the first 10 mixed spectra
    #     plt.plot(Raman_Shift, Intensity_mix[i, :], label=f'Mixed Spectrum {i+1} (Label: {Label_mix[i]})')
    # plt.xlabel('Raman Shift (cm⁻¹)')
    # plt.ylabel('Intensity')
    # plt.title(f'Digitally Mixed SERS Spectra for {CA} Identification')
    # plt.legend()
    # plt.show()

    return Intensity_mix, Label_mix


def plot_probability_distributions_by_label(probabilities, labels, title, folders):
    """Plot probability distributions split by binary labels for each molecule."""
    molecules = ['DA', 'E', 'NE']
    colors = {0: 'tab:blue', 1: 'tab:orange'}
    offsets = {0: -0.18, 1: 0.18}

    fig, ax = plt.subplots(figsize=(11, 6))

    for idx, molecule in enumerate(molecules, start=1):
        probs = np.asarray(probabilities[molecule]).reshape(-1)
        labs = np.asarray(labels[molecule]).reshape(-1)

        for group in (0, 1):
            group_probs = probs[labs == group]
            if group_probs.size == 0:
                continue

            position = idx + offsets[group]
            ax.boxplot(
                group_probs,
                positions=[position],
                widths=0.28,
                patch_artist=True,
                boxprops=dict(facecolor=colors[group], alpha=0.25, color=colors[group]),
                medianprops=dict(color=colors[group], linewidth=2),
                whiskerprops=dict(color=colors[group]),
                capprops=dict(color=colors[group]),
                flierprops=dict(markeredgecolor=colors[group], markerfacecolor=colors[group], alpha=0.5),
            )

            jitter = np.random.normal(position, 0.03, size=group_probs.shape[0])
            ax.scatter(
                jitter,
                group_probs,
                color=colors[group],
                alpha=0.55,
                s=18,
                label=f'Label {group}' if idx == 1 else None,
            )

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(molecules)
    ax.set_xlabel('Molecule')
    ax.set_ylabel('Predicted Probability')
    ax.set_title(title)
    ax.legend(title='True label')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    plt.savefig('visualization/Probability distribution.png', dpi = 600)
    plt.show(block = False)
    plt.pause(5)
    plt.close()

    # plot probability distributions for each folder, 3 subplots for DA, E, NE
    plt.figure(figsize=(12, 20))

    for idx, molecule in enumerate(molecules, start=1):
        plt.subplot(3, 1, idx)
        # plot box plot, hue by folder
        df = pd.DataFrame({
            'Probability': probabilities[molecule].reshape(-1),
            'Label': labels[molecule].reshape(-1),
            'folder': folders
        })
        sns.boxplot(x='folder', y='Probability', data=df)
        sns.stripplot(x='folder', y='Probability', data=df, alpha=0.5, jitter=True)
        plt.title(f'Probability distribution for {molecule}')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualization/Probability distribution by folder.png', dpi = 600)
    plt.show(block = False)
    plt.pause(5)
    plt.close()