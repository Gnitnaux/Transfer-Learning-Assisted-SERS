"""
Prediction module for Random Forest Identification Model.
"""

from src.utils import read_spectra_predict
from src.utils import spectra_normalization
from src.model import RF_Identification_Predict

def test_Identification_Model(data_dir, model_dir):
    """
    Test the Random Forest Identification Model using SERS data for prediction.
    
    Args:
        data_dir (str): Path to the preprocessed data directory for prediction.
        model_dir (str): Path to the trained model directory.
    """
    # Load and preprocess prediction data
    Raman_Shift, Intensity, Concentrations = read_spectra_predict(data_dir)
    print(f"Raman Shift shape: {Raman_Shift.shape}")
    print(f"Intensity shape: {Intensity.shape}")
    print("Prediction data loaded successfully.")

    # Data normalization
    Intensity_norm = spectra_normalization(Raman_Shift, Intensity, 
                                           peak_position=1082, peak_range=30, plot=True)
    print("Prediction data normalization completed.")

    # Load and test Identification Model (Model 1)
    DA_Labels = (Concentrations[:, 0] > 0).astype(int)  # 1 if DA present, else 0
    DA_Predictions, DA_Probabilities = RF_Identification_Predict(Raman_Shift, Intensity_norm, 'DA', model_dir, plot=True, labels=DA_Labels)
    E_Labels = (Concentrations[:, 1] > 0).astype(int)  # 1 if E present, else 0
    E_Predictions, E_Probabilities = RF_Identification_Predict(Raman_Shift, Intensity_norm, 'E', model_dir, plot=True, labels=E_Labels)
    NE_Labels = (Concentrations[:, 2] > 0).astype(int)  # 1 if NE present, else 0
    NE_Predictions, NE_Probabilities = RF_Identification_Predict(Raman_Shift, Intensity_norm, 'NE', model_dir, plot=True, labels=NE_Labels)
    print("Identification models tested successfully.")