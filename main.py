#!/usr/bin/env python3
"""
Main program interface for Transfer Learning Assisted SERS
This script serves as the main entry point for the SERS analysis pipeline.
"""

import argparse
import sys
from pathlib import Path

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
        choices=["preprocess", "train", "evaluate", "predict"],
        default="train",
        help="Operation mode: preprocess, train, evaluate, or predict"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to raw data directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/preprocessed",
        help="Path to output directory for preprocessed data"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Path to model directory"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Transfer Learning Assisted SERS")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model directory: {args.model_dir}")
    print("=" * 60)
    
    if args.mode == "preprocess":
        from preprocess import run_preprocessing
        print("\nRunning preprocessing...")
        run_preprocessing(args.data_dir, args.output_dir)
        
    elif args.mode == "train":
        print("\nTraining mode selected.")
        print("Training functionality to be implemented in src/ directory")
        # TODO: Import and call training function from src/
        
    elif args.mode == "evaluate":
        print("\nEvaluation mode selected.")
        print("Evaluation functionality to be implemented in src/ directory")
        # TODO: Import and call evaluation function from src/
        
    elif args.mode == "predict":
        print("\nPrediction mode selected.")
        print("Prediction functionality to be implemented in src/ directory")
        # TODO: Import and call prediction function from src/
    
    print("\nDone!")


if __name__ == "__main__":
    main()
