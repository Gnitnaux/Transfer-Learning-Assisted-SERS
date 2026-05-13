#!/usr/bin/env python3
"""
Main program interface for Transfer Learning Assisted SERS
This script serves as the main entry point for the SERS analysis pipeline.
"""

import argparse
import sys
import os
from pathlib import Path
from src.train import train_model
from src.predict import test_Identification_Model
from src.train import test_train_ratio_model
from src.predict import test_Ratio_Model
from src.predict import Ratio_prediction_test
from src.plsr_model import PLSR_Train, PLSR_Predict
from src.utils import read_spectra_train, read_spectra_test, spectra_normalization
import numpy as np

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Main function to orchestrate the SERS analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Transfer Learning Assisted SERS - Main Program Interface"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict", "test", "test_IdModel", "test_RatioModel_train",
                 "test_RatioModel_predict", "plsr_train", "plsr_test"],
        default="train",
        help="Operation mode: train, predict, test, plsr_train, or plsr_test"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/preprocessed",
        help="Path to preprocessed data directory"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Path to model directory"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Transfer Learning Assisted SERS")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Data directory: {args.data_dir}")
    print(f"Model directory: {args.model_dir}")
    print("=" * 60)
    
    if args.mode == "train":
        print("\nTraining mode selected.")
        train_model(os.path.join(args.data_dir, 'train'), args.model_dir)

        
    elif args.mode == "predict":
        print("\nPrediction mode selected.")
        print("Prediction functionality to be implemented in src/ directory")
        # TODO: Import and call prediction function from src/
    
    elif args.mode == "test":
        print("\nTesting mode selected.")
        Ratio_prediction_test(os.path.join(args.data_dir, 'test'), args.model_dir)

    elif args.mode == "test_IdModel":
        print("\nTesting Identification Model mode selected.")
        test_Identification_Model(os.path.join(args.data_dir, 'test'), args.model_dir)

    elif args.mode == "test_RatioModel_train":
        print("\nTesting Ratio Model on training data mode selected.")
        test_train_ratio_model(os.path.join(args.data_dir, 'train'), args.model_dir)

    elif args.mode == "test_RatioModel_predict":
        # the test is on train set
        print("\nTesting Ratio Model on prediction data mode selected.")
        test_Ratio_Model(os.path.join(args.data_dir, 'train'), args.model_dir)

    elif args.mode == "plsr_train":
        print("\nPLSR Training mode selected.")
        Raman_Shift, Intensity, Category, Concentration = read_spectra_train(
            os.path.join(args.data_dir, 'train')
        )
        print(f"Raman Shift shape: {Raman_Shift.shape}")
        print(f"Intensity shape: {Intensity.shape}")
        Intensity_norm = spectra_normalization(
            Raman_Shift, Intensity,
            peak_position=920, peak_range=20, plot=True, mode='plsr_train', minmax_scale = False
        )
        print("Data normalization completed.")
        PLSR_Train(Raman_Shift, Intensity_norm, Concentration, Category,
                   args.model_dir, plot=True)
        print("PLSR model trained successfully.")

    elif args.mode == "plsr_test":
        print("\nPLSR Testing mode selected.")
        Raman_Shift, Intensity, Concentrations = read_spectra_test(
            os.path.join(args.data_dir, 'test')
        )
        print(f"Raman Shift shape: {Raman_Shift.shape}")
        print(f"Intensity shape: {Intensity.shape}")
        Intensity_norm = spectra_normalization(
            Raman_Shift, Intensity,
            peak_position=920, peak_range=20, plot=True, mode='plsr_test', minmax_scale = False
        )
        print("Data normalization completed.")
        total_true = np.sum(Concentrations, axis=1)
        predictions = PLSR_Predict(Intensity_norm, args.model_dir,
                                   plot=True, true_concentrations=total_true)
        rmse = np.sqrt(np.mean((predictions - total_true) ** 2))
        print(f"PLSR Test Results:")
        print(f"  RMSE: {rmse:.4f} uM")
        for i, (true_val, pred_val) in enumerate(zip(total_true, predictions)):
            print(f"  Sample {i}: True={true_val:.2f} uM, Pred={pred_val:.2f} uM")

    print("\nDone!")


if __name__ == "__main__":
    main()
