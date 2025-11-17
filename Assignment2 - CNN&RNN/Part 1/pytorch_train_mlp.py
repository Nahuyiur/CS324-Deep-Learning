from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from pytorch_mlp import MLP
import torch
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None

def accuracy(predictions, labels):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    predictions = np.argmax(predictions, axis=1)
    labels = np.argmax(labels, axis=1)
    accuracy = np.mean(predictions == labels)
    return accuracy

def train():
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    model = MLP(FLAGS.n_inputs, FLAGS.dnn_hidden_units, FLAGS.n_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    best_accuracy = 0
    for epoch in range(FLAGS.max_steps):
        model.train()
        for i, (inputs, labels) in enumerate(train_data):
            optimizer.zero_grad()   
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step()
            if i % FLAGS.eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    outputs = model(inputs)
                    accuracy = accuracy(outputs, labels)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        torch.save(model.state_dict(), 'best_model.pth')
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            accuracy = accuracy(outputs, labels)


def main():
    """
    Main function
    """
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()