"""
Prediction module for Random Forest Identification Model.
"""

from src.utils import read_spectra_train
from src.utils import read_spectra_test
from src.utils import read_spectra_unknown
from src.utils import spectra_normalization
from src.utils import plot_probability_distributions_by_label
from src.model import RF_Identification_Predict
from src.model import RF_Ratio_Predict
from src.ae_id_model import AE_Identification_Predict_Multi
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def test_Identification_Model(data_dir, model_dir):
    """
    Test the Random Forest Identification Model using SERS data for prediction.
    
    Args:
        data_dir (str): Path to the preprocessed data directory for prediction.
        model_dir (str): Path to the trained model directory.
    """
    # Load and preprocess prediction data
    Raman_Shift, Intensity, Folders = read_spectra_unknown(data_dir)
    print(f"Raman Shift shape: {Raman_Shift.shape}")
    print(f"Intensity shape: {Intensity.shape}")
    print("Prediction data loaded successfully.")

    # Data normalization
    Intensity_norm = spectra_normalization(Raman_Shift, Intensity, 
                                           peak_position=920, peak_range=20, plot=True, mode = 'test_IDModel')
    print("Prediction data normalization completed.")

    # extract concentrations from folders, the order of CAs in folder is DA, E, NE
    Concentration_list = []
    for folder in Folders:
            DA_con = (folder.split('_')[0]).split('u')[0]
            E_con = (folder.split('_')[1]).split('u')[0]
            NE_con = (folder.split('_')[2]).split('u')[0]
            Concentration_list.append([float(DA_con), float(E_con), float(NE_con)]) 

    Concentrations = np.array(Concentration_list, dtype=float)

    # Load and test Identification Model (Model 1)
    DA_Labels = (Concentrations[:, 0] > 0).astype(int)  # 1 if DA present, else 0
    E_Labels = (Concentrations[:, 1] > 0).astype(int)  # 1 if E present, else 0
    NE_Labels = (Concentrations[:, 2] > 0).astype(int)  # 1 if NE present, else 0
    multi_predictions, multi_probabilities = AE_Identification_Predict_Multi(Intensity_norm, model_dir)
    DA_Predictions, E_Predictions, NE_Predictions = multi_predictions[:, 0], multi_predictions[:, 1], multi_predictions[:, 2]
    DA_Probabilities, E_Probabilities, NE_Probabilities = (
        multi_probabilities[:, 0],
        multi_probabilities[:, 1],
        multi_probabilities[:, 2],
    )

    for molecule, labels, predictions in [
        ('DA', DA_Labels, DA_Predictions),
        ('E', E_Labels, E_Predictions),
        ('NE', NE_Labels, NE_Predictions),
    ]:
        cm = confusion_matrix(labels, predictions, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'Not {molecule}', molecule])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for {molecule} Identification (AE)')
        plt.savefig(f'visualization/AE_Identification_{molecule}_Confusion_Matrix.png', dpi=600)
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    print("Identification models tested successfully.")

    # RF_Identification_Predict returns class-1 probabilities as a 1D array.
    DA_probs = np.asarray(DA_Probabilities).reshape(-1)
    E_probs = np.asarray(E_Probabilities).reshape(-1)
    NE_probs = np.asarray(NE_Probabilities).reshape(-1)

    plot_probability_distributions_by_label(
        probabilities={'DA': DA_probs, 'E': E_probs, 'NE': NE_probs},
        labels={'DA': DA_Labels, 'E': E_Labels, 'NE': NE_Labels},
        title='Predicted Probability Distributions for DA, E, and NE by True Label',
        folders = Folders
    )

    # use expectation for folder label prediction, if probability > 0.5, predict present, else predict absent
    folder_predictions = []
    folder_true_label = []
    folder_probabilities = []
    for folder in np.unique(Folders):
        indices = [i for i, f in enumerate(Folders) if f == folder]
        DA_prob_mean = np.mean(DA_probs[indices])
        E_prob_mean = np.mean(E_probs[indices])
        NE_prob_mean = np.mean(NE_probs[indices])

        DA_pred = 1 if DA_prob_mean > 0.5 else 0
        E_pred = 1 if E_prob_mean > 0.5 else 0
        NE_pred = 1 if NE_prob_mean > 0.5 else 0

        folder_predictions.append((folder, DA_pred, E_pred, NE_pred))
        folder_true_label.append((folder, DA_Labels[indices[0]], E_Labels[indices[0]], NE_Labels[indices[0]]))
        folder_probabilities.append((folder, DA_prob_mean, E_prob_mean, NE_prob_mean))

    # plot probabilities for each folder in a bar plot, 3 subplots for DA, E, NE
    plt.figure(figsize=(12, 6))
    for idx, molecule in enumerate(['DA', 'E', 'NE'], start=1):
        plt.subplot(1, 3, idx)
        folder_names = [fp[0] for fp in folder_probabilities]
        prob_means = [fp[idx] for fp in folder_probabilities]
        # orange for true label 1, blue for true label 0, red dashed line for threshold
        colors = ['orange' if folder_true_label[i][idx] == 1 else 'blue' for i in range(len(folder_true_label))]
        
        plt.bar(folder_names, prob_means, color=colors)
        plt.axhline(0.5, color='red', linestyle='--')
        plt.title(f'Mean Predicted Probability for {molecule} by Folder')
        plt.xlabel('Folder')
        plt.ylabel('Mean Predicted Probability')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualization/Mean_Predicted_Probabilities_by_Folder.png', dpi = 600)
    plt.show(block = False)
    plt.pause(5)
    plt.close()


def test_Ratio_Model(data_dir, model_dir):
    """
    Test the transfer learning ratio model using SERS data.
    
    Args:
        data_dir (str): Path to the preprocessed data directory.
        model_dir (str): Path to the trained model directory.
    """
    # Load and preprocess training data
    Raman_Shift, Intensity, Category, Concentration = read_spectra_train(data_dir)
    print(f"Raman Shift shape: {Raman_Shift.shape}")
    print(f"Intensity shape: {Intensity.shape}")
    print("Data loaded successfully.")

    # Data normalization
    Intensity_norm = spectra_normalization(Raman_Shift, Intensity,
                                           peak_position=920, peak_range=20, plot=True, mode = 'test_RatioModel')
    print("Data normalization completed.")

    # Load and test Ratio Model (Model 2), the test is not to measure ratio but category prediction
    CAs = ['DA', 'E']
    Intensity_norm_filtered = Intensity_norm[np.isin(Category, CAs)]
    Category_filtered = Category[np.isin(Category, CAs)]
    Ratio_Predictions_DA_E, Ratio_Probabilities_DA_E = RF_Ratio_Predict(Intensity_norm_filtered, CAs, model_dir, plot=True, labels=Category_filtered)

    CAs = ['DA', 'NE']
    Intensity_norm_filtered = Intensity_norm[np.isin(Category, CAs)]
    Category_filtered = Category[np.isin(Category, CAs)]
    Ratio_Predictions_DA_NE, Ratio_Probabilities_DA_NE = RF_Ratio_Predict(Intensity_norm_filtered, CAs, model_dir, plot=True, labels=Category_filtered)

    CAs = ['E', 'NE']
    Intensity_norm_filtered = Intensity_norm[np.isin(Category, CAs)]
    Category_filtered = Category[np.isin(Category, CAs)]
    Ratio_Predictions_E_NE, Ratio_Probabilities_E_NE = RF_Ratio_Predict(Intensity_norm_filtered, CAs, model_dir, plot=True, labels=Category_filtered)

    CAs = ['DA', 'E', 'NE']
    Intensity_norm_filtered = Intensity_norm[np.isin(Category, CAs)]
    Category_filtered = Category[np.isin(Category, CAs)]
    Ratio_Predictions_DA_E_NE, Ratio_Probabilities_DA_E_NE = RF_Ratio_Predict(Intensity_norm_filtered, CAs, model_dir, plot=True, labels=Category_filtered)

    print("Ratio models tested successfully.")


def Ratio_prediction_test(data_dir, model_dir):
    """
    Predict concentration ratios using the trained Ratio Model.
    
    Args:
        data_dir (str): Path to the preprocessed data directory for prediction.
        model_dir (str): Path to the trained model directory.
    Returns:
        list: Predicted concentration ratios [DA_ratio, E_ratio, NE_ratio].
    """
    # Load and preprocess prediction data
    Raman_Shift, Intensity, Concentrations = read_spectra_test(data_dir)
    print(f"Raman Shift shape: {Raman_Shift.shape}")
    print(f"Intensity shape: {Intensity.shape}")
    print("Prediction data loaded successfully.")

    # Data normalization
    Intensity_norm = spectra_normalization(Raman_Shift, Intensity, 
                                           peak_position=920, peak_range=20, plot=True, mode = 'testRatio_predict')
    print("Prediction data normalization completed.")

    ratio_pred = []
    ratio_real = []
    label_real = []
    label_pred = []

    # a dictionary for label mapping
    label_map = {0x000: 'None', 0x100: 'DA', 0x010: 'E', 0x001: 'NE',
                 0x110: 'DA+E', 0x101: 'DA+NE', 0x011: 'E+NE', 0x111: 'DA+E+NE'}

    for concentration in Concentrations:
        ratio_real.append(concentration / np.sum(concentration) if np.sum(concentration) > 0 else [0,0,0])
        label = 0x000
        if concentration[0] > 0:
            label += 0x100
        if concentration[1] > 0:
            label += 0x010
        if concentration[2] > 0:
            label += 0x001
        label_real.append(label_map[label])
        

    # Step 1 - Identify present molecules using Identification Models
    DA_Labels = (Concentrations[:, 0] > 0).astype(int)  # 1 if DA present, else 0
    E_Labels = (Concentrations[:, 1] > 0).astype(int)  # 1 if E present, else 0
    NE_Labels = (Concentrations[:, 2] > 0).astype(int)  # 1 if NE present, else 0
    multi_predictions, multi_probabilities = AE_Identification_Predict_Multi(Intensity_norm, model_dir)
    DA_Predictions, E_Predictions, NE_Predictions = multi_predictions[:, 0], multi_predictions[:, 1], multi_predictions[:, 2]
    DA_Probabilities, E_Probabilities, NE_Probabilities = (
        multi_probabilities[:, 0],
        multi_probabilities[:, 1],
        multi_probabilities[:, 2],
    )


    # Step 2 - Predict concentration ratios using Ratio Models
    last_con = []
    for i, concentration in enumerate(Concentrations):
        if len(last_con) == 0:
            print(f"Predicting ratios for [DA, E, NE] concentrations: {concentration}")
        elif not np.array_equal(concentration, last_con):
            print(f"Predicting ratios for [DA, E, NE] concentrations: {concentration}")

        present_CAs = []
        label = 0x000
        if DA_Predictions[i] > 0:
            present_CAs.append('DA')
            label += 0x100
        if E_Predictions[i] > 0:
            present_CAs.append('E')
            label += 0x010
        if NE_Predictions[i] > 0:
            present_CAs.append('NE')
            label += 0x001
        label_pred.append(label_map[label])
        
        if len(present_CAs) == 0:
            ratio_pred.append([0, 0, 0])
            print(f"Real ratio:{ratio_real[i]}, Predicted ratio:[0, 0, 0]")

        elif len(present_CAs) == 1:
            if present_CAs[0] == 'DA':
                ratio_pred.append([1, 0, 0])
                print(f"Real ratio:{ratio_real[i]}, Predicted ratio:[1, 0, 0]")
            elif present_CAs[0] == 'E':
                ratio_pred.append([0, 1, 0])
                print(f"Real ratio:{ratio_real[i]}, Predicted ratio:[0, 1, 0]")
            elif present_CAs[0] == 'NE':
                ratio_pred.append([0, 0, 1])
                print(f"Real ratio:{ratio_real[i]}, Predicted ratio:[0, 0, 1]")

        else:
            Single_Intensity = Intensity_norm[i].reshape(1, -1)
            Ratio_Predictions, Ratio_Probabilities = RF_Ratio_Predict(Single_Intensity, present_CAs, model_dir, plot=False)
            # append probability as ratio
            prob = Ratio_Probabilities[0] / np.sum(Ratio_Probabilities[0])

            if 'DA' in present_CAs:
                DA_index = present_CAs.index('DA')
                DA_ratio = prob[DA_index]
            else:
                DA_ratio = 0

            if 'E' in present_CAs:
                E_index = present_CAs.index('E')
                E_ratio = prob[E_index]
            else:
                E_ratio = 0

            if 'NE' in present_CAs:
                NE_index = present_CAs.index('NE')
                NE_ratio = prob[NE_index]
            else:
                NE_ratio = 0

            ratio_pred.append([DA_ratio, E_ratio, NE_ratio])
            print(f"Real ratio:{ratio_real[i]}, Predicted ratio:[{DA_ratio:.3f}, {E_ratio:.3f}, {NE_ratio:.3f}]")

        last_con = concentration

    # calculate average prediction for each unique concentration in Concentrations
    print("\nCalculating average predicted ratios for each unique concentration...")
    unique_concentrations = np.unique(Concentrations, axis=0)
    unique_real_ratios = []
    avg_ratio_pred = []
    for unique_con in unique_concentrations:
        indices = [i for i, con in enumerate(Concentrations) if np.array_equal(con, unique_con)]
        unique_real_ratios.append(ratio_real[indices[0]])
        avg_pred = np.mean([ratio_pred[i] for i in indices], axis=0)
        avg_ratio_pred.append(avg_pred)
        print(f"Average predicted ratio for concentration {unique_con}: [{avg_pred[0]:.3f}, {avg_pred[1]:.3f}, {avg_pred[2]:.3f}]")

    # calculate RMSE between avg_ratio_pred and unique_real_ratios
    print("\nCalculating RMSE between average predicted ratios and real ratios...")
    unique_real_ratios = np.array(unique_real_ratios)
    avg_ratio_pred = np.array(avg_ratio_pred)
    rmse = np.sqrt(np.mean((unique_real_ratios - avg_ratio_pred) ** 2, axis=0))
    print(f"\nRMSE between average predicted ratios and real ratios: DA: {rmse[0]:.3f}, E: {rmse[1]:.3f}, NE: {rmse[2]:.3f}")

    # plot confusion matrix for identification results
    cm = confusion_matrix(label_real, label_pred, labels=list(label_map.values()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.values()))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title('Confusion Matrix for Molecule Identification')
    plt.tight_layout()
    plt.savefig('visualization/Ratio_predction_test-Identification_Confusion_Matrix.png', dpi = 600)
    plt.show(block = False)
    plt.pause(5)
    plt.close()
        
    # plot pred-real pairs in 3D scatter plot, line the pair togther
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(unique_real_ratios[:, 0], unique_real_ratios[:, 1], unique_real_ratios[:, 2], c='b', marker='o', label='Real Ratios')
    ax.scatter(avg_ratio_pred[:, 0], avg_ratio_pred[:, 1], avg_ratio_pred[:, 2], c='r', marker='^', label='Predicted Ratios')
    for i in range(len(unique_real_ratios)):
        ax.plot([unique_real_ratios[i, 0], avg_ratio_pred[i, 0]],
                [unique_real_ratios[i, 1], avg_ratio_pred[i, 1]],
                [unique_real_ratios[i, 2], avg_ratio_pred[i, 2]], 'k--', linewidth=0.5)
    ax.set_xlabel('DA Ratio')
    ax.set_ylabel('E Ratio')
    ax.set_zlabel('NE Ratio')
    ax.set_title('Predicted vs Real Concentration Ratios')
    ax.legend()
    plt.savefig('visualization/Ratio_predction_test-Predicted_vs_Real_Ratios.png', dpi = 600)
    plt.show(block = False)
    plt.pause(5)
    plt.close()

    print("Ratio prediction completed.")
