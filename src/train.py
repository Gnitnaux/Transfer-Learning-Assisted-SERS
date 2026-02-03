"""
Training module for Transfer Learning Assisted SERS analysis
"""

from src.utils import read_spectra_train
from src.utils import spectra_normalization

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
                                           peak_position=1480, peak_range=20, plot=True)
    print("Data normalization completed.")

    # Build Identification Model (Model 1)