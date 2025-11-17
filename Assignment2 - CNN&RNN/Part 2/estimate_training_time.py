"""
Estimate training time for CNN on M4 Pro MacBook
"""
import torch
import torch.nn as nn
import time
from cnn_model import CNN

# Training configuration
BATCH_SIZE = 32
MAX_STEPS = 5000
CIFAR10_TRAIN_SIZE = 50000  # CIFAR10 has 50,000 training images

# Check device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model
model = CNN(n_channels=3, n_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Create dummy data (batch_size=32, 3 channels, 32x32 images)
dummy_input = torch.randn(BATCH_SIZE, 3, 32, 32).to(device)
dummy_labels = torch.randint(0, 10, (BATCH_SIZE,)).to(device)

# Warm up
print("Warming up...")
for _ in range(10):
    optimizer.zero_grad()
    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_labels)
    loss.backward()
    optimizer.step()

# Time a few iterations
print("Timing training iterations...")
num_test_iterations = 50
start_time = time.time()

for i in range(num_test_iterations):
    optimizer.zero_grad()
    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_labels)
    loss.backward()
    optimizer.step()

elapsed_time = time.time() - start_time
avg_time_per_step = elapsed_time / num_test_iterations

# Calculate total training time
total_steps = MAX_STEPS
total_training_time = avg_time_per_step * total_steps

# Also estimate evaluation time (full dataset pass)
print("\nEstimating evaluation time...")
model.eval()
eval_start = time.time()
with torch.no_grad():
    # Simulate evaluation: forward pass only (no backward)
    for _ in range(50):  # Use more batches for better estimate
        outputs = model(dummy_input)
        # Simulate accuracy calculation
        _ = outputs.max(1)
eval_time_per_batch = (time.time() - eval_start) / 50

# CIFAR10 train set: 50,000 images / 32 batch_size = ~1563 batches
# CIFAR10 test set: 10,000 images / 32 batch_size = ~313 batches
train_eval_batches = 1563
test_eval_batches = 313
total_eval_batches = train_eval_batches + test_eval_batches
eval_time_per_evaluation = total_eval_batches * eval_time_per_batch

# Number of evaluations during training
num_evals = (MAX_STEPS // 500) + 1  # eval every 500 steps
total_eval_time = eval_time_per_evaluation * num_evals

# Total time
total_time = total_training_time + total_eval_time

print("\n" + "="*60)
print("TRAINING TIME ESTIMATION")
print("="*60)
print(f"Device: {device}")
print(f"Average time per training step: {avg_time_per_step:.4f} seconds")
print(f"Number of training steps: {total_steps:,}")
print(f"Training time (steps only): {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours)")
print(f"\nEvaluation batches: {total_eval_batches} (train: {train_eval_batches}, test: {test_eval_batches})")
print(f"Evaluation time per evaluation: {eval_time_per_evaluation:.2f} seconds ({eval_time_per_evaluation/60:.2f} minutes)")
print(f"Number of evaluations: {num_evals}")
print(f"Total evaluation time: {total_eval_time/60:.2f} minutes ({total_eval_time/3600:.2f} hours)")
print(f"\n{'='*60}")
print(f"ESTIMATED TOTAL TIME: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
print(f"{'='*60}")

