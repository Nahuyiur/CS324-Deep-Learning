# CS324 Assignment 2: CNN & RNN

This assignment implements three deep learning models using PyTorch: Multilayer Perceptron (MLP), Convolutional Neural Network (CNN), and Recurrent Neural Network (RNN).

## Environment Setup

### Prerequisites
- Python 3.7+
- pip

### Installation

```bash
# Install PyTorch (CPU version)
pip install torch torchvision

# Install other dependencies
pip install numpy matplotlib scikit-learn jupyter notebook
```

For GPU support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/).

## Project Structure

```
Assignment2 - CNN&RNN/
├── Part 1/                    # MLP Implementation
│   ├── pytorch_mlp.py        # PyTorch MLP model
│   ├── cifar10_mlp.py        # CIFAR-10 MLP model
│   ├── pytorch_train_mlp.py  # Training script
│   ├── task2_compare_mlp.ipynb    # NumPy vs PyTorch comparison
│   ├── task3_cifar10.ipynb        # CIFAR-10 classification
│   └── data/                 # CIFAR-10 dataset (auto-downloaded)
│
├── Part 2/                   # CNN Implementation
│   ├── cnn_model.py          # CNN architecture
│   ├── cnn_train.py          # Training script
│   └── cnn_analysis.ipynb     # Performance analysis
│
├── Part 3/                   # RNN Implementation
│   ├── vanilla_rnn.py        # Vanilla RNN from scratch
│   ├── train.py              # Training script
│   ├── dataset.py            # Palindrome dataset
│   └── palindrome_rnn_analysis.ipynb  # Length analysis
│
├── images/                   # Figures for report
├── report.tex                # LaTeX report
├── compile.py                # LaTeX compilation script
└── README.md                 # This file
```

## Usage

### Part 1: MLP

**Task 2 - Compare NumPy and PyTorch:**
```bash
cd Part\ 1
jupyter notebook task2_compare_mlp.ipynb
```

**Task 3 - CIFAR-10 Classification:**
```bash
cd Part\ 1
jupyter notebook task3_cifar10.ipynb
```

### Part 2: CNN

**Train CNN:**
```bash
cd Part\ 2
python cnn_train.py --early_stopping --patience 5 --min_delta 0.05
```

**Analysis:**
```bash
cd Part\ 2
jupyter notebook cnn_analysis.ipynb
```

### Part 3: RNN

**Train RNN:**
```bash
cd Part\ 3
python train.py --input_length 10 --num_hidden 128 --train_steps 500
```

**Analysis:**
```bash
cd Part\ 3
jupyter notebook palindrome_rnn_analysis.ipynb
```

### Compile Report

```bash
python compile.py
```

This generates `report.pdf` from `report.tex`.

## Key Features

- **Part 1**: MLP implementation with NumPy/PyTorch comparison and CIFAR-10 classification
- **Part 2**: VGG-style CNN with batch normalization and early stopping
- **Part 3**: Vanilla RNN from scratch (no built-in modules) for palindrome prediction

## Notes

- CIFAR-10 dataset will be automatically downloaded to `Part 1/data/` on first run
- All models use default hyperparameters as specified in the assignment
- Training logs and checkpoints are saved during training

