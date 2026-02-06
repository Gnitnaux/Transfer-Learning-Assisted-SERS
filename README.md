# Transfer-Learning-Assisted-SERS

Transfer learning strategy for catecholamine quantification in mixture, motivated by Emily et al. [1] And the author of this project is Xuanting Liu from Shanghai Jiao Tong University, with the assistant of Microsoft Copilot.

[1] E. X. Tan, J. R. T. Chen, D. W. C. Pang, N. S. Tan, I. Y. Phang, X. Y. Ling, *Angew. Chem. Int. Ed*. 2025, 64, e202508717. https://doi.org/10.1002/anie.202508717

## Project Structure

```
Transfer-Learning-Assisted-SERS/
│
├── data/                       # Data storage directory
│   ├── raw/                    # Raw SERS data files
│   │   ├── train/              # Training data subfolders
│   │   │   ├── DA_1uM_1/        # Each subfolder contains CSV files
│   │   │   ├── DA_2uM_1/        # with [Raman Shift, Intensity] format
│   │   │   └── ...
│   │   └── test/               # Test data subfolders
│   │       ├── 2uM_6uM_8uM_1/        # Same structure as train/
│   │       ├── 1uM_3uM_10uM_1/
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
│   ├── train.py               # Functions for training
|   ├── predict.py             # Functions for rediction
|   └── utils.py               # Utility functions
|
├── visualization/             # Visualized png storage
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

### Data Preparation

The folder name (label) in train should be `[DA/E/NE]_[XX]uM_[Index]`, specially, name the background 'BA_0uM_Index'
The folder name (label) in test should be `[DA]uM_[E]uM_[NE]uM_[Index]`

prepare all the data as the structure below


```
data/raw/
├── train/                      # Training dataset
│   ├── DA_1uM_1/                 # Each class/category has its own folder
│   │   ├── sample1.csv
│   │   ├── sample2.csv
│   │   └── ...
│   ├── DA_2uM_1/
│   │   └── ...
│   └── ...
└── test/                       # Test dataset
    ├── 2uM_6uM_8uM_1/                 # Same structure as train
    │   ├── sample1.csv
    │   └── ...
    └── ...
```

Each CSV file should contain two columns:
- Column 1: Raman Shift (wavelength values)
- Column 2: Intensity (spectral intensity values)

**Important Notes**:
- The first row will be skipped during processing (assumed to be headers)
- Files should use either GBK encoding (if there is any Chinese)
- All spectra in the same subfolder should have the same Raman shift range

### Work Pipeline

The `main.py` script serves as the main entry point for the SERS analysis pipeline.

```bash
# Preprocessing
# before all the procedure, store the raw data in data/raw
# run it to preprocess raw data in train and test seperately 
# save the processed data in data/preprocessed
python preprocess.py 

# Run training
python main.py --mode train --data-dir data/preprocessed --model-dir models

# Run prediction
python main.py --mode predict --model-dir models
```

### Preprocessing Program

The `preprocess.py` script can be run independently for data preprocessing.
It applies SG (Savitzky-Golay) filtering and AirPLS baseline correction to both training and test datasets. It's a GUI program which is easy to use.



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
- **\_\_init\_\_.py**: Package initialization

### visualization
Store the visualizitions generated while running with `.png` format


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms specified in the LICENSE file.

## Contact

For questions or feedback, please open an issue on GitHub.

