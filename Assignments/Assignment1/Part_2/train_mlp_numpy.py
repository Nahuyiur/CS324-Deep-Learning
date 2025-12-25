import argparse
import numpy as np
from mlp_numpy import MLP  
from modules import CrossEntropy,Linear
from sklearn.datasets import make_moons
from sklearn.utils import shuffle
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500 # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10

N_SAMPLES = 1000
NOISE = 0.1
TRAIN_RATIO = 0.8

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    # TODO: Implement the accuracy calculation
    # Hint: Use np.argmax to find predicted classes, and compare with the true classes in targets
    pred_cls = np.argmax(predictions, axis=1)
    true_cls = np.argmax(targets, axis=1)
    return (pred_cls == true_cls).mean() * 100.0

def parse_hidden_units(dnn_hidden_units: str):
    if dnn_hidden_units is None:
        return []
    s = dnn_hidden_units.strip()
    if not s:
        return []
    return [int(x) for x in s.split(',') if x.strip()]

def generate_datapoints(n_samples=N_SAMPLES,noise=NOISE):
    X,y=make_moons(n_samples=n_samples,noise=noise)
    X,y=shuffle(X,y)
    return X,y

def get_batches(X, y, batch_size):
    """
    Generator that yields mini-batches of data.
    """
    n_samples = X.shape[0]

    if batch_size is None or batch_size >= n_samples:
        # Full batch gradient descent
        yield X, y
    else:
        # Mini-batch or stochastic gradient descent
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]

def train(dnn_hidden_units, learning_rate, max_steps, eval_freq, batch_size=None, return_history=False):
    """
    Performs training and evaluation of MLP model.

    Args:
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        batch_size: Batch size for mini-batch gradient descent.
                    If None, use full batch gradient descent.
                    If 1, use stochastic gradient descent.
        return_history: If True, return training history
        NOTE: Add necessary arguments such as the data, your model...
    """
    # TODO: Load your data here
    X,y=generate_datapoints()

    n_classes=int(y.max())+1
    y_onehot=np.eye(n_classes)[y] #(N,C)

    split=int(len(X)*TRAIN_RATIO)
    X_train,y_train=X[:split],y_onehot[:split]
    X_test,y_test=X[split:],y_onehot[split:]

    hidden_units = parse_hidden_units(dnn_hidden_units)
    model = MLP(n_inputs=X.shape[1], n_hidden=hidden_units, n_classes=n_classes)
    # TODO: Initialize your MLP model and loss function (CrossEntropy) here

    # Track history
    history = {
        'steps': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    for step in range(max_steps):
        # TODO: Implement the training loop with mini-batch support
        epoch_losses = []

        # Iterate over mini-batches
        for batch_X, batch_y in get_batches(X_train, y_train, batch_size):
            # 1. Forward pass
            # 2. Compute loss
            loss, probs = model.forward(batch_X, batch_y)
            epoch_losses.append(loss)

            # 3. Backward pass (compute gradients)
            model.backward(None, batch_y)

            # 4. Update weights
            for layer in model.layers:
                if isinstance(layer, Linear):
                    layer.params['weight'] -= learning_rate * layer.grads['weight']
                    layer.params['bias']   -= learning_rate * layer.grads['bias']

        # Average loss over all batches in this epoch
        avg_loss = np.mean(epoch_losses)

        if step % eval_freq == 0 or step == max_steps - 1:
            # TODO: Evaluate the model on the test set
            # 1. Forward pass on the test set
            # 2. Compute loss and accuracy
            # Evaluate on full training set
            train_loss, train_probs = model.forward(X_train, y_train)
            train_acc = accuracy(train_probs, y_train)

            # Evaluate on test set
            probs_test = model.forward(X_test)
            test_ce = CrossEntropy()
            test_loss = test_ce.forward(probs_test, y_test)
            test_acc = accuracy(probs_test, y_test)

            # Record history
            if return_history:
                history['steps'].append(step)
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['test_loss'].append(test_loss)
                history['test_acc'].append(test_acc)

            print(f"Step: {step:4d} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:5.2f}%")

    print("Training complete!")

    if return_history:
        return history

def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS = parser.parse_known_args()[0]
    
    train(FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq)

if __name__ == '__main__':
    main()
