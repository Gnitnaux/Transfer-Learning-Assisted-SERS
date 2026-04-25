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
from src.ae_unmixing_model import AE_Unmixing_Predict
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
    Test the direct AE unmixing model on the real mixed test dataset.
    
    Args:
        data_dir (str): Path to the preprocessed data directory for prediction.
        model_dir (str): Path to the trained model directory.
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

    prediction_payload = AE_Unmixing_Predict(Intensity_norm, model_dir)
    concentration_pred = prediction_payload["concentrations"]
    ratio_pred = prediction_payload["ratios"]
    ratio_real = np.divide(
        Concentrations,
        np.sum(Concentrations, axis=1, keepdims=True) + 1e-8,
        out=np.zeros_like(Concentrations, dtype=float),
        where=np.sum(Concentrations, axis=1, keepdims=True) > 0,
    )

    # average predictions for each unique concentration setting
    unique_concentrations = np.unique(Concentrations, axis=0)
    avg_concentration_pred = []
    avg_ratio_pred = []
    unique_ratio_real = []
    for unique_con in unique_concentrations:
        indices = np.where(np.all(Concentrations == unique_con, axis=1))[0]
        avg_concentration_pred.append(np.mean(concentration_pred[indices], axis=0))
        avg_ratio_pred.append(np.mean(ratio_pred[indices], axis=0))
        unique_ratio_real.append(ratio_real[indices[0]])
        print(
            f"Condition {unique_con}: "
            f"pred concentration = {np.mean(concentration_pred[indices], axis=0)}, "
            f"pred ratio = {np.mean(ratio_pred[indices], axis=0)}"
        )

    avg_concentration_pred = np.asarray(avg_concentration_pred, dtype=float)
    avg_ratio_pred = np.asarray(avg_ratio_pred, dtype=float)
    unique_ratio_real = np.asarray(unique_ratio_real, dtype=float)

    concentration_rmse = np.sqrt(np.mean((avg_concentration_pred - unique_concentrations) ** 2, axis=0))
    ratio_rmse = np.sqrt(np.mean((avg_ratio_pred - unique_ratio_real) ** 2, axis=0))
    print(
        "Average concentration RMSE: "
        f"DA={concentration_rmse[0]:.3f}, E={concentration_rmse[1]:.3f}, NE={concentration_rmse[2]:.3f}"
    )
    print(
        "Average ratio RMSE: "
        f"DA={ratio_rmse[0]:.3f}, E={ratio_rmse[1]:.3f}, NE={ratio_rmse[2]:.3f}"
    )

    def _plot_scatter(actual, predicted, labels, title, path):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for idx, molecule in enumerate(labels):
            ax = axes[idx]
            ax.scatter(actual[:, idx], predicted[:, idx], s=40, alpha=0.8)
            lower = float(min(np.min(actual[:, idx]), np.min(predicted[:, idx])))
            upper = float(max(np.max(actual[:, idx]), np.max(predicted[:, idx])))
            if np.isclose(lower, upper):
                upper = lower + 1.0
            ax.plot([lower, upper], [lower, upper], "k--", linewidth=1.5)
            ax.set_xlabel(f"Actual {molecule}")
            ax.set_ylabel(f"Predicted {molecule}")
            ax.set_title(molecule)
            ax.grid(True, alpha=0.3)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=600)
        plt.show(block=False)
        plt.pause(5)
        plt.close(fig)

    _plot_scatter(
        unique_concentrations,
        avg_concentration_pred,
        ["DA Concentration", "E Concentration", "NE Concentration"],
        "AE Unmixing: Predicted vs Actual Concentration",
        "visualization/AE_Unmixing_Test_Concentration_Scatter.png",
    )
    _plot_scatter(
        unique_ratio_real,
        avg_ratio_pred,
        ["DA Ratio", "E Ratio", "NE Ratio"],
        "AE Unmixing: Predicted vs Actual Ratio",
        "visualization/AE_Unmixing_Test_Ratio_Scatter.png",
    )

    print("AE unmixing test completed.")
