# Transfer-Learning-Assisted-SERS

Transfer learning strategy for catecholamine quantification in mixture, motivated by Angew. Chem. Int. Ed. 2025, 64, e202508717 (doi.org/10.1002/anie).202508717

## Project Structure

```
Transfer-Learning-Assisted-SERS/
│
├── data/                       # Data storage directory
│   ├── raw/                    # Raw SERS data files
│   │   ├── train/              # Training data subfolders
│   │   │   ├── folder1/        # Each subfolder contains CSV files
│   │   │   ├── folder2/        # with [Raman Shift, Intensity] format
│   │   │   └── ...
│   │   └── test/               # Test data subfolders
│   │       ├── folder1/        # Same structure as train/
│   │       ├── folder2/
│   │       └── ...
│   └── preprocessed/           # Preprocessed data files
│       ├── train/              # Preprocessed training data
│       └── test/               # Preprocessed test data
│
├── models/                     # Trained model storage
│
├── src/                        # Source code directory
│   ├── __init__.py            # Package initialization
│   ├── model.py               # Model definitions and training functions
│   └── utils.py               # Utility functions
│
├── main.py                     # Main program interface
├── preprocess.py              # Data preprocessing program
├── .gitignore                 # Git ignore file
├── LICENSE                    # License file
└── README.md                  # This file
```

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Gnitnaux/Transfer-Learning-Assisted-SERS.git
cd Transfer-Learning-Assisted-SERS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start Example

Here's a complete workflow from data organization to preprocessing:

```bash
# 1. Organize your data
# Place your CSV files in the following structure:
# data/raw/train/class1/sample1.csv
# data/raw/train/class1/sample2.csv
# data/raw/train/class2/sample1.csv
# data/raw/test/class1/sample1.csv
# data/raw/test/class2/sample1.csv

# 2. Run preprocessing with default parameters
python preprocess.py

# 3. Check the output
# Preprocessed files will be in:
# data/preprocessed/train/class1/processed_sample1.csv
# data/preprocessed/train/class1/processed_class1_mean.csv
# data/preprocessed/test/class1/processed_sample1.csv
# etc.
```

### Data Organization

Before preprocessing, organize your data as follows:

```
data/raw/
├── train/                      # Training dataset
│   ├── class1/                 # Each class/category has its own folder
│   │   ├── sample1.csv
│   │   ├── sample2.csv
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
└── test/                       # Test dataset
    ├── class1/                 # Same structure as train
    │   ├── sample1.csv
    │   └── ...
    └── ...
```

Each CSV file should contain two columns:
- Column 1: Raman Shift (wavelength values)
- Column 2: Intensity (spectral intensity values)

**Important Notes**:
- The first row will be skipped during processing (assumed to be headers)
- Files should use either GBK or UTF-8 encoding
- All spectra in the same subfolder should have the same Raman shift range
- Typical Raman shift range: 400-1800 cm⁻¹ (configurable)

### Main Program Interface

The `main.py` script serves as the main entry point for the SERS analysis pipeline.

```bash
# Run preprocessing (processes both train and test folders)
python main.py --mode preprocess --data-dir data/raw --output-dir data/preprocessed

# Run training
python main.py --mode train --data-dir data/preprocessed --model-dir models

# Run prediction
python main.py --mode predict --model-dir models
```

### Preprocessing Program

The `preprocess.py` script can be run independently for data preprocessing.
It applies SG (Savitzky-Golay) filtering and AirPLS baseline correction to both 
training and test datasets:

```bash
# Basic usage with default parameters
python preprocess.py --data-dir data/raw --output-dir data/preprocessed

# Advanced usage with custom parameters
python preprocess.py \
    --data-dir data/raw \
    --output-dir data/preprocessed \
    --window-length 7 \
    --polyorder 3 \
    --lambda-val 1000000 \
    --porder 3 \
    --prefix processed_
```

**Preprocessing Parameters:**
- `--window-length`: SG filter window length (must be odd, default: 7)
- `--polyorder`: SG filter polynomial order (default: 3)
- `--lambda-val`: AirPLS lambda parameter (default: 1e6)
- `--porder`: AirPLS polynomial order (default: 3)
- `--prefix`: Prefix for output filenames (default: "processed_")

### Available Options

- `--mode`: Operation mode (preprocess, train, evaluate, predict)
- `--data-dir`: Path to data directory
- `--output-dir`: Path to output directory
- `--model-dir`: Path to model directory
- `--config`: Path to configuration file (optional)

## Directory Descriptions

### data/
- **raw/**: Store your raw SERS data files here
  - **train/**: Training dataset organized in subfolders by class/category
  - **test/**: Test dataset organized in subfolders by class/category
  - Each CSV file should have 2 columns: `[Raman Shift, Intensity]`
- **preprocessed/**: Preprocessed and normalized data will be saved here
  - **train/**: Preprocessed training data (after SG filtering and AirPLS baseline correction)
  - **test/**: Preprocessed test data

### models/
Store trained models here. Models will be saved with descriptive names including timestamps and performance metrics.

### src/
Contains all custom functions and detailed training code:
- **model.py**: Transfer learning model definitions, training, and evaluation functions
- **utils.py**: Utility functions for data validation, metrics calculation, and configuration management
- **__init__.py**: Package initialization

## Development

### Adding New Features

1. Add custom functions to the `src/` directory
2. Update `main.py` or create new scripts to use these functions
3. Document your changes in the code and README

### Data Processing Pipeline

The preprocessing pipeline applies the following steps to each spectral file:

1. **SG Filtering**: Savitzky-Golay filter smooths spectral data
   - Window length: adjustable (default: 7)
   - Polynomial order: adjustable (default: 3)
   - Reduces high-frequency noise while preserving peak shapes

2. **AirPLS Baseline Correction**: Adaptive Iteratively Reweighted Penalized Least Squares
   - Lambda: adjustable (default: 1e6) - controls smoothness of baseline
   - Polynomial order: adjustable (default: 3)
   - Automatically removes baseline drift and background fluorescence

**Algorithm Details**: The processing uses a reference-based approach where spectra are 
processed alongside their Raman shift values to ensure consistent scaling and alignment.
This implementation follows the algorithm from the original spectroscopy processing GUI.

The processing is applied independently to:
- All files in `data/raw/train/` subfolders → saved to `data/preprocessed/train/`
- All files in `data/raw/test/` subfolders → saved to `data/preprocessed/test/`

Each processed subfolder also includes a merged CSV file containing the mean spectrum across all samples in that folder.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms specified in the LICENSE file.

## Contact

For questions or feedback, please open an issue on GitHub.

