"""
Model definitions and training functions for Transfer Learning
"""

def RF_Identification_Train(Intensity, Category, CA, model_dir):
    """
    Train Random Forest model to identify pure CA from blank SERS spectra.
    Args:
        Intensity (np.ndarray): 2D array of intensity values (samples x features).
        Category (np.ndarray): Array of category labels for each sample.
        CA (str): Target Molecule.
        model_dir (str): Directory to save the trained model.
    """