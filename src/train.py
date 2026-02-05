"""
Training module for Transfer Learning Assisted SERS analysis
"""

from src.utils import read_spectra_train
from src.utils import spectra_normalization
from src.model import RF_Identification_Train
from src.model import RF_Ratio_Train

def train_model(data_dir, model_dir):
    """
    Train the transfer learning model using SERS data.
    
    Args:
        data_dir (str): Path to the preprocessed data directory.
        model_dir (str): Path to save the trained model.
    """
    # Load and preprocess training data
    Raman_Shift, Intensity, Category, Concentration = read_spectra_train(data_dir)
    print(f"Raman Shift shape: {Raman_Shift.shape}")
    print(f"Intensity shape: {Intensity.shape}")
    print("Data loaded successfully.")

    # Data normalization
    Intensity_norm = spectra_normalization(Raman_Shift, Intensity, 
                                           peak_position=1082, peak_range=30, plot=True)
    print("Data normalization completed.")

    # Build Identification Model (Model 1)
    DA_IdModel = RF_Identification_Train(Raman_Shift, Intensity_norm, Category, 'DA', model_dir, plot=True)
    E_IdModel = RF_Identification_Train(Raman_Shift, Intensity_norm, Category, 'E', model_dir, plot=True)
    NE_IdModel = RF_Identification_Train(Raman_Shift, Intensity_norm, Category, 'NE', model_dir, plot=True)
    print("Identification models trained successfully.")

    # Build Ratio Model (Model 2)
    CAs = ['DA', 'E']
    RatioModel_DA_E = RF_Ratio_Train(Raman_Shift, Intensity_norm, Category, CAs, model_dir, plot=True)

    CAs = ['DA', 'NE']
    RatioModel_DA_NE = RF_Ratio_Train(Raman_Shift, Intensity_norm, Category, CAs, model_dir, plot=True)
    
    CAs = ['E', 'NE']
    RatioModel_E_NE = RF_Ratio_Train(Raman_Shift, Intensity_norm, Category, CAs, model_dir, plot=True)

    CAs = ['DA', 'E', 'NE']
    RatioModel_DA_E_NE = RF_Ratio_Train(Raman_Shift, Intensity_norm, Category, CAs, model_dir, plot=True) 

    print("Ratio model trained successfully.")


def test_train_ratio_model(data_dir, model_dir):
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
                                           peak_position=1082, peak_range=30, plot=True)
    print("Data normalization completed.")

    # Build Ratio Model (Model 2)
    CAs = ['DA', 'E']
    RatioModel_DA_E = RF_Ratio_Train(Raman_Shift, Intensity_norm, Category, CAs, model_dir, plot=True)

    CAs = ['DA', 'NE']
    RatioModel_DA_NE = RF_Ratio_Train(Raman_Shift, Intensity_norm, Category, CAs, model_dir, plot=True)
    
    CAs = ['E', 'NE']
    RatioModel_E_NE = RF_Ratio_Train(Raman_Shift, Intensity_norm, Category, CAs, model_dir, plot=True)

    CAs = ['DA', 'E', 'NE']
    RatioModel_DA_E_NE = RF_Ratio_Train(Raman_Shift, Intensity_norm, Category, CAs, model_dir, plot=True) 

    print("Ratio model trained successfully.")