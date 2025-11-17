from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN

def train(config):

    # Initialize the model that we are going to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_interval = getattr(config, 'log_interval', 10)
    model = VanillaRNN(
        seq_length=config.input_length,
        input_dim=config.input_dim,
        hidden_dim=config.num_hidden,
        output_dim=config.num_classes,
        batch_size=config.batch_size
    ).to(device)

    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    final_loss = None
    final_accuracy = None

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        batch_inputs = batch_inputs.to(device=device, dtype=torch.float32)
        if batch_inputs.dim() == 2:
            batch_inputs = batch_inputs.unsqueeze(-1)
        batch_targets = batch_targets.to(device=device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(batch_inputs)
        loss = criterion(logits, batch_targets)
        loss.backward()

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == batch_targets).float().mean().item()
        loss_value = loss.item()
        final_loss = loss_value
        final_accuracy = accuracy

        if step % log_interval == 0 or step == config.train_steps:
            # print acuracy/loss here
            print('Step {:05d} | Loss {:.4f} | Accuracy {:.4f}'.format(step, loss_value, accuracy))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    return model, final_loss, final_accuracy

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--log_interval', type=int, default=10, help='Steps between logging metrics')

    config = parser.parse_args()
    # Train the model
    train(config)