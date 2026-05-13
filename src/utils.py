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
ID_MOLECULES = ['DA', 'E', 'NE']

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

def spectra_normalization(Raman_Shift, Intensity, peak_position = 1480, peak_range = 20, plot = False, mode = 'train', minmax_scale = True):
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
    if minmax_scale:
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


def digital_mix_ID(Raman_Shift, Intensity, Category, Concentration, CA, num_mix=1000, Range=(0.5, 10)):
    """
    Create digital mixed spectra for identification model training.

    Uses single-component spectra from ALL available concentrations (not just
    the highest). When generating a mixture, each component's spectrum is
    drawn from the closest available concentration.

    Args:
        Raman_Shift (np.ndarray): Array of Raman shift values.
        Intensity (np.ndarray): 2D array of intensity values (samples x features).
        Category (np.ndarray): Array of category labels for each sample.
        Concentration (np.ndarray): Array of concentration values for each sample.
        CA (str): The target chemical agent for identification.
        num_mix (int): Number of mixed spectra to generate.
        Range (list): The range of total mixture concentrations.
    Returns:
        Intensity_mix (np.ndarray): 2D array of digitally mixed intensity values.
        Label_mix (0/1): Array of binary labels indicating presence (1) or absence (0)
            of the target CA.
    """
    _ = Raman_Shift

    rng = np.random.default_rng(42)
    category = np.asarray(Category)
    concentration = np.asarray(Concentration, dtype=float)

    spectra_pool = {}
    concentration_levels = {}
    for molecule in ['DA', 'E', 'NE', 'BA']:
        indices = np.where(category == molecule)[0]
        if indices.size == 0:
            raise ValueError(f"Category '{molecule}' is required for digital mixing.")
        spectra_pool[molecule] = {}
        conc_values = np.unique(concentration[indices])
        concentration_levels[molecule] = sorted(conc_values)
        for conc in conc_values:
            conc_indices = indices[concentration[indices] == conc]
            spectra_pool[molecule][conc] = np.asarray(
                Intensity[conc_indices], dtype=np.float32
            )

    max_conc = max(concentration_levels['DA'])

    mix_concentration = rng.uniform(Range[0], Range[1], num_mix)
    ratio_CA = mix_concentration / max_conc

    ratio_CAs = []
    Label_mix = []

    for i in range(num_mix // 2):
        ratio_target = (rng.uniform(0.3, 1, 1)[0]) * ratio_CA[i]
        ratio_other_1 = (rng.uniform(0, 1 - ratio_target, 1)[0]) * ratio_CA[i]
        ratio_other_2 = (1 - ratio_target - ratio_other_1) * ratio_CA[i]
        if CA == 'DA':
            ratio_CAs.append([ratio_target, ratio_other_1, ratio_other_2, 1 - ratio_CA[i]])
        elif CA == 'E':
            ratio_CAs.append([ratio_other_1, ratio_target, ratio_other_2, 1 - ratio_CA[i]])
        elif CA == 'NE':
            ratio_CAs.append([ratio_other_1, ratio_other_2, ratio_target, 1 - ratio_CA[i]])
        Label_mix.append(1)

    for i in range(num_mix - num_mix // 2):
        ratio_other_1 = (rng.uniform(0, 1, 1)[0]) * ratio_CA[i]
        ratio_other_2 = (rng.uniform(0, 1 - ratio_other_1, 1)[0]) * ratio_CA[i]
        if CA == 'DA':
            ratio_CAs.append([0, ratio_other_1, ratio_other_2, 1 - ratio_CA[i]])
        elif CA == 'E':
            ratio_CAs.append([ratio_other_1, 0, ratio_other_2, 1 - ratio_CA[i]])
        elif CA == 'NE':
            ratio_CAs.append([ratio_other_1, ratio_other_2, 0, 1 - ratio_CA[i]])
        Label_mix.append(0)

    Intensity_mix = []
    molecules = ['DA', 'E', 'NE']
    for i in range(num_mix):
        spectrum = np.zeros(Intensity.shape[1], dtype=np.float32)
        total_conc = mix_concentration[i]
        spectrum += ratio_CAs[i][3] * _sample_spectrum(
            spectra_pool['BA'], concentration_levels['BA'], 0.0, rng
        )
        for j, mol in enumerate(molecules):
            target_conc = ratio_CAs[i][j] * max_conc
            if ratio_CAs[i][j] > 0:
                spectrum += ratio_CAs[i][j] * _sample_spectrum(
                    spectra_pool[mol], concentration_levels[mol], target_conc, rng
                )
        Intensity_mix.append(spectrum)

    Intensity_mix = np.array(Intensity_mix)
    Label_mix = np.array(Label_mix)

    return Intensity_mix, Label_mix


def digital_mix_ID_multilabel(Raman_Shift, Intensity, Category, Concentration, samples_per_combination=400, Range=(0.5, 10.0), seed=42):
    """
    Create digitally mixed spectra for a shared multi-label identification model.

    Uses single-component spectra from ALL available concentrations (not just 10 uM).
    When generating a mixture, each component's spectrum is drawn from the closest
    available concentration to the target concentration in the mixture. This yields
    more realistic spectral shapes than using only a single reference concentration.

    Eight balanced mixture patterns are generated:
    BA, DA, E, NE, DA+E, DA+NE, E+NE, DA+E+NE.

    Args:
        Raman_Shift (np.ndarray): Array of Raman shift values.
        Intensity (np.ndarray): 2D array of intensity values (samples x features).
        Category (np.ndarray): Array of category labels for each sample.
        Concentration (np.ndarray): Array of concentration values for each sample.
        samples_per_combination (int): Number of synthetic spectra per composition pattern.
        Range (tuple): Total analyte concentration range in uM for non-background mixtures.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple:
            Intensity_mix (np.ndarray): Synthetic mixed spectra.
            Label_mix (np.ndarray): Binary presence labels with shape (N, 3) for DA/E/NE.
            Abundance_mix (np.ndarray): Mixing ratios with shape (N, 4) for DA/E/NE/BA.
            Combination_labels (np.ndarray): Integer labels in [0, 7].
    """
    _ = Raman_Shift

    # normalize the spectrum without min-max scaling
    Intensity_norm = spectra_normalization(Raman_Shift, Intensity, peak_position=920, peak_range=20, 
                                           plot=False, mode = 'digital_mix', minmax_scale = False)

    rng = np.random.default_rng(seed)
    category = np.asarray(Category)
    concentration = np.asarray(Concentration, dtype=float)

    # Build (category, concentration) → spectra pool
    spectra_pool = {}
    concentration_levels = {}
    for molecule in ['DA', 'E', 'NE', 'BA']:
        indices = np.where(category == molecule)[0]
        if indices.size == 0:
            raise ValueError(f"Category '{molecule}' is required for digital mixing.")
        spectra_pool[molecule] = {}
        conc_values = np.unique(concentration[indices])
        concentration_levels[molecule] = sorted(conc_values)
        for conc in conc_values:
            conc_indices = indices[concentration[indices] == conc]
            spectra_pool[molecule][conc] = np.asarray(
                Intensity_norm[conc_indices], dtype=np.float32
            )

    max_conc = max(concentration_levels['DA'])

    combinations = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]

    mixed_spectra = []
    binary_labels = []
    abundance_labels = []
    combination_labels = []

    for combination_id, combination in enumerate(combinations):
        active_indices = [idx for idx, present in enumerate(combination) if present]
        for _sample_idx in range(samples_per_combination):
            if active_indices:
                total_concentration = rng.uniform(Range[0], Range[1])
                total_ratio = total_concentration / max_conc
                component_split = rng.dirichlet(np.ones(len(active_indices)))
            else:
                total_ratio = 0.0
                component_split = np.array([], dtype=np.float32)

            abundance = np.zeros(4, dtype=np.float32)
            for split_value, active_idx in zip(component_split, active_indices):
                abundance[active_idx] = total_ratio * float(split_value)
            abundance[3] = max(0.0, 1.0 - float(np.sum(abundance[:3])))

            spectrum = np.zeros(Intensity_norm.shape[1], dtype=np.float32)
            spectrum += abundance[3] * _sample_spectrum(
                spectra_pool['BA'], concentration_levels['BA'], 0.0, rng
            )
            for active_idx in active_indices:
                molecule = ['DA', 'E', 'NE'][active_idx]
                target_conc = float(total_concentration * component_split[active_indices.index(active_idx)])
                spectrum += abundance[active_idx] * _sample_spectrum(
                    spectra_pool[molecule], concentration_levels[molecule], target_conc, rng
                )

            mixed_spectra.append(spectrum)
            binary_labels.append(np.asarray(combination, dtype=np.float32))
            abundance_labels.append(abundance)
            combination_labels.append(combination_id)

    # min-max scaling to [0, 1]
    mixed_spectra = np.array(mixed_spectra)
    min_vals = np.min(mixed_spectra, axis=1, keepdims=True)
    max_vals = np.max(mixed_spectra, axis=1, keepdims=True)
    mixed_spectra = (mixed_spectra - min_vals) / (max_vals - min_vals + 1e-8)

    return (
        np.asarray(mixed_spectra, dtype=np.float32),
        np.asarray(binary_labels, dtype=np.float32),
        np.asarray(abundance_labels, dtype=np.float32),
        np.asarray(combination_labels, dtype=np.int64),
    )


def _sample_spectrum(spectra_by_conc, concentration_levels, target_conc, rng):
    """Select a random spectrum from the concentration level closest to target_conc."""
    if target_conc <= 0 or len(concentration_levels) == 1:
        conc_key = concentration_levels[0]
    else:
        # find closest concentration level
        diffs = [abs(c - target_conc) for c in concentration_levels]
        conc_key = concentration_levels[int(np.argmin(diffs))]
    pool = spectra_by_conc[conc_key]
    return pool[rng.integers(len(pool))]


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
