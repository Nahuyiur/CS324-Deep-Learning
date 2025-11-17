from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from cnn_model import CNN

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = '../Part 1/data'

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    if len(targets.shape) > 1 and targets.shape[1] > 1:
        targets = np.argmax(targets, axis=1)
    
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    
    return np.mean(predictions == targets)

def train(learning_rate=LEARNING_RATE_DEFAULT, batch_size=BATCH_SIZE_DEFAULT, 
          max_steps=MAX_EPOCHS_DEFAULT, eval_freq=EVAL_FREQ_DEFAULT, 
          data_dir=DATA_DIR_DEFAULT, return_history=False):
    """
    Performs training and evaluation of CNN model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Check if dataset already exists
    if os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')):
        print("CIFAR10 dataset found, skipping download...")
        download = False
    else:
        print("CIFAR10 dataset not found, downloading...")
        download = True
    
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=download, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=download, transform=transform_test)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model = CNN(n_channels=3, n_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {'steps': [], 'train_loss': [], 'train_acc': [], 'test_acc': []} if return_history else None
    
    for epoch in range(max_steps):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if (epoch + 1) % eval_freq == 0 or epoch == max_steps - 1:
            model.eval()
            train_correct = 0
            train_total = 0
            test_correct = 0
            test_total = 0
            train_loss_sum = 0.0
            
            with torch.no_grad():
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    train_loss_sum += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()
                
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * train_correct / train_total
            test_acc = 100. * test_correct / test_total
            avg_loss = train_loss_sum / len(trainloader)
            
            if return_history:
                history['steps'].append(epoch + 1)
                history['train_loss'].append(avg_loss)
                history['train_acc'].append(train_acc)
                history['test_acc'].append(test_acc)
            
            print(f'Epoch [{epoch+1}/{max_steps}]')
            print(f'Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    if return_history:
        return model, history
    return model

def main():
    """
    Main function
    """
    train(learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size,
          max_steps=FLAGS.max_steps, eval_freq=FLAGS.eval_freq, data_dir=FLAGS.data_dir)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
