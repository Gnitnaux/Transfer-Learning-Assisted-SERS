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
        choices=["train", "predict", "test", "test_IdModel", "test_RatioModel_train", "test_RatioModel_predict"],
        default="train",
        help="Operation mode: train or predict"
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
        
    print("\nDone!")


if __name__ == "__main__":
    main()
