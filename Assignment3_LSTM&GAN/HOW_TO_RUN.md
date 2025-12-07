# How to Run Assignment 3 Code

This document provides step-by-step instructions on how to run the code for both **Part I: LSTM** and **Part II: GAN**.

## Environment Setup

Ensure you have the required dependencies installed. You can install them via pip:

```bash
pip install torch torchvision matplotlib numpy
```

## Part I: PyTorch LSTM

The code for this part is located in the `Part 1/` directory.

### 1. Training and Evaluation

To train the LSTM model on the default palindrome task (length=19), simply run:

```bash
cd "Part 1"
python train.py
```

This will:
- Train the LSTM model for 100 epochs (default).
- Print training and validation loss/accuracy.
- Use the settings defined in `train.py` (Hidden dim: 128, Batch size: 128, LR: 0.001).

### 2. Running the Analysis (Task 2 & 3)

To reproduce the analysis and the comparison with Vanilla RNN (including the T=5 perfect accuracy test and the length sweep), run the provided Jupyter notebook:

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook palindrome_lstm_analysis.ipynb
   ```
2. Run all cells in the notebook.
   - This will train the LSTM on palindrome lengths [5, 10, 15, 20, 25, 30].
   - It will generate the `lstm_vs_rnn_comparison.png` plot used in the report.

---

## Part II: Generative Adversarial Networks (GANs)

The code for this part is located in the `Part 2/` directory.

### 1. Training the GAN (Task 1)

To train the GAN on the MNIST dataset:

```bash
cd "Part 2"
python my_gan.py --n_epochs 100
```

- **Arguments:**
  - `--n_epochs`: Number of training epochs (default is 100, as used in the report).
  - `--batch_size`: Batch size (default: 64).
  - `--lr`: Learning rate (default: 0.0002).
  - `--save_interval`: Interval for saving generated image samples (default: 500 batches).

- **Outputs:**
  - Generated images will be saved in the `images/` folder (e.g., `0.png`, `93500.png` etc.).
  - The trained generator model will be saved as `mnist_generator.pt`.

### 2. Generating Samples and Interpolation (Task 2 & 3)

To generate the specific samples and interpolation figures for the report, use the provided Jupyter notebook:

1. Ensure you have trained the model first (so `mnist_generator.pt` exists), or the notebook will fail to load the model.
2. Start Jupyter Notebook:
   ```bash
   jupyter notebook gan_analysis.ipynb
   ```
3. Run the cells to:
   - **Task 2**: Display samples from the start, middle, and end of training.
   - **Task 3**: Perform latent space interpolation between two digits.
     - This will generate and save `interpolation_result.png`.

