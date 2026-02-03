# Transfer-Learning-Assisted-SERS

Transfer learning strategy for catecholamine quantification in mixture, motivated by Angew. Chem. Int. Ed. 2025, 64, e202508717 (doi.org/10.1002/anie).202508717

## Project Structure

```
Transfer-Learning-Assisted-SERS/
│
├── data/                       # Data storage directory
│   ├── raw/                    # Raw SERS data files
│   └── preprocessed/           # Preprocessed data files
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

### Main Program Interface

The `main.py` script serves as the main entry point for the SERS analysis pipeline.

```bash
# Run preprocessing
python main.py --mode preprocess --data-dir data/raw --output-dir data/preprocessed

# Run training
python main.py --mode train --data-dir data/preprocessed --model-dir models

# Run prediction
python main.py --mode predict --model-dir models
```

### Preprocessing Program

The `preprocess.py` script can be run independently for data preprocessing:

```bash
python preprocess.py --data-dir data/raw --output-dir data/preprocessed
```

### Available Options

- `--mode`: Operation mode (preprocess, train, evaluate, predict)
- `--data-dir`: Path to data directory
- `--output-dir`: Path to output directory
- `--model-dir`: Path to model directory
- `--config`: Path to configuration file (optional)

## Directory Descriptions

### data/
- **raw/**: Store your raw SERS data files here (CSV format, with 2 columns `[Raman shift, Intensity]`)
- **preprocessed/**: Preprocessed and normalized data will be saved here

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



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms specified in the LICENSE file.

## Contact

For questions or feedback, please open an issue on GitHub.

