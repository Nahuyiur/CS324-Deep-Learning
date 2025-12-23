# CS324 Deep Learning - Assignment 1

This repository contains the implementation of Assignment 1, covering three parts: Perceptron, Multi-Layer Perceptron, and Stochastic Gradient Descent.

## Directory Structure

```
Assignment1/
├── Part_1/
│   ├── perceptron.py          # Perceptron implementation
│   ├── part1.ipynb             # Experiments and results for Part I
│   └── perceptron_tutorial.pdf
├── Part_2/
│   ├── modules.py              # Neural network modules (Linear, ReLU, Softmax, etc.)
│   ├── mlp_numpy.py            # MLP architecture
│   ├── train_mlp_numpy.py      # Training script with batch/SGD support
│   ├── part2.ipynb             # Part II experiments (Batch Gradient Descent)
│   └── part3.ipynb             # Part III experiments (SGD and batch size analysis)
├── report.tex                  # LaTeX report
└── README.md                   # This file
```

## Requirements

```bash
Python >= 3.7
numpy
matplotlib
scikit-learn
jupyter
```

Install dependencies:
```bash
pip install numpy matplotlib scikit-learn jupyter
```

## How to Run

### Part I: Perceptron

1. Open and run the Jupyter notebook:
```bash
cd Part_1
jupyter notebook part1.ipynb
```

2. Alternatively, use the perceptron module directly:
```python
from perceptron import Perceptron
import numpy as np

# Create perceptron
p = Perceptron(n_inputs=2, max_epochs=1000, learning_rate=0.1)

# Train
p.train(X_train, y_train)

# Predict
predictions = p.forward(X_test)
```

### Part II: Multi-Layer Perceptron (Batch Gradient Descent)

1. Run the Jupyter notebook for Part II:
```bash
cd Part_2
jupyter notebook part2.ipynb
```

2. Or run training from command line with default parameters:
```bash
cd Part_2
python train_mlp_numpy.py
```

3. Customize hyperparameters:
```bash
python train_mlp_numpy.py --dnn_hidden_units 20 --learning_rate 0.01 --max_steps 1500 --eval_freq 10
```

**Available arguments:**
- `--dnn_hidden_units`: Comma-separated hidden layer sizes (default: '20')
- `--learning_rate`: Learning rate (default: 0.01)
- `--max_steps`: Number of training epochs (default: 1500)
- `--eval_freq`: Evaluation frequency (default: 10)

### Part III: Stochastic Gradient Descent

1. Run the Jupyter notebook for Part III:
```bash
cd Part_2
jupyter notebook part3.ipynb
```

2. Or run from command line with batch size parameter:

**SGD (batch_size=1):**
```bash
python -c "
from train_mlp_numpy import train
history = train(
    dnn_hidden_units='20',
    learning_rate=0.01,
    max_steps=1500,
    eval_freq=10,
    batch_size=1,  # SGD
    return_history=True
)
"
```

**Mini-batch GD (batch_size=32):**
```bash
python -c "
from train_mlp_numpy import train
history = train(
    dnn_hidden_units='20',
    learning_rate=0.01,
    max_steps=1500,
    eval_freq=10,
    batch_size=32,  # Mini-batch
    return_history=True
)
"
```

**Full Batch GD (batch_size=None):**
```bash
python train_mlp_numpy.py  # Default is full batch
```

## Code Structure

### Part I: Perceptron (`Part_1/perceptron.py`)

**Key Methods:**
- `__init__(n_inputs, max_epochs, learning_rate)`: Initialize perceptron
- `forward(input)`: Compute predictions
- `train(training_inputs, labels)`: Train using perceptron learning rule

### Part II: MLP Modules (`Part_2/modules.py`)

**Implemented Modules:**
- `Linear`: Fully connected layer with forward/backward passes
- `ReLU`: ReLU activation function
- `SoftMax`: Softmax activation with numerical stability
- `CrossEntropy`: Cross-entropy loss

### Part II: MLP Architecture (`Part_2/mlp_numpy.py`)

**MLP Class:**
- `__init__(n_inputs, n_hidden, n_classes)`: Build network architecture
- `forward(x, y=None)`: Forward pass through all layers
- `backward(dout, y)`: Backpropagation through all layers

### Part II & III: Training (`Part_2/train_mlp_numpy.py`)

**Main Function:**
- `train(dnn_hidden_units, learning_rate, max_steps, eval_freq, batch_size=None, return_history=False)`
  - `batch_size=None`: Full batch gradient descent (Part II)
  - `batch_size=1`: Stochastic gradient descent (Part III)
  - `batch_size=N`: Mini-batch gradient descent (Part III)

**Helper Functions:**
- `generate_datapoints()`: Generate moon dataset
- `get_batches()`: Create mini-batches for training
- `accuracy()`: Compute classification accuracy

## Expected Results

### Part I: Perceptron
- Test accuracy: ~80% on well-separated Gaussian distributions
- Performance degrades when means are too close or variances too high

### Part II: MLP (Batch GD)
- **Default parameters (hidden_units=20, lr=0.01, epochs=1500):**
  - Training Accuracy: ~82%
  - Test Accuracy: ~83%
  - Final Training Loss: ~0.40
  - Final Test Loss: ~0.40

### Part III: SGD Analysis
- **SGD (batch_size=1):**
  - Best test accuracy: ~93%
  - Fastest convergence
  - More noise in training

- **Mini-batch (batch_size=8-32):**
  - Good balance: ~87-91% test accuracy
  - Stable and efficient

- **Large batch (batch_size=128, Full):**
  - Slower convergence with lr=0.01
  - Test accuracy: ~82-85%

## Notebooks

All Jupyter notebooks contain:
- Complete experiments with visualizations
- Training/test accuracy curves
- Loss curves
- Detailed analysis and observations

**To view all results:**
```bash
# Part I
jupyter notebook Part_1/part1.ipynb

# Part II (Batch GD)
jupyter notebook Part_2/part2.ipynb

# Part III (SGD analysis)
jupyter notebook Part_2/part3.ipynb
```

## Report

The detailed report is available in `report.tex`. To compile:

```bash
pdflatex report.tex
pdflatex report.tex  # Run twice for references
```

Or use your preferred LaTeX editor (Overleaf, TeXShop, etc.).

## Notes

- All random seeds are set in the notebooks for reproducibility
- The make_moons dataset has noise=0.2 by default
- Train/test split is 80/20 for all parts
- All implementations use only NumPy (no PyTorch/TensorFlow)

## Author

Rui Yuhan
Student Number: 12310520
Course: CS324 Deep Learning
Date: October 2025
