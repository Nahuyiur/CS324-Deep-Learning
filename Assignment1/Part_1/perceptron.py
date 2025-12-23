import numpy as np

class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=200, learning_rate=0.01):
        """
        Initializes the perceptron object.
        - n_inputs: Number of inputs.
        - max_epochs: Maximum number of training cycles.
        - learning_rate: Magnitude of weight changes at each training cycle.
        - weights: Initialize weights (including bias).
        """
        self.n_inputs = n_inputs  # Fill in: Initialize number of inputs
        self.max_epochs = max_epochs  # Fill in: Initialize maximum number of epochs
        self.learning_rate = learning_rate  # Fill in: Initialize learning rate
        self.weights = np.zeros(n_inputs+1)  # Fill in: Initialize weights with zeros 
        
    def forward(self, input_vec):
        """
        Predicts label from input.
        Args:
            input_vec (np.ndarray): Input array of training data, input vec must be all samples
        Returns:
            int: Predicted label (1 or -1) or Predicted lables.
        """
        X = np.asarray(input_vec, dtype=float)
        input_was_1d = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            input_was_1d = True

        ones=np.ones((X.shape[0], 1), dtype=float)
        X_aug=np.concatenate([X, ones], axis=1)   # (n, d+1)

        scores = np.zeros(X_aug.shape[0], dtype=float)
        for i in range(X_aug.shape[0]):
            scores[i] = float(np.dot(X_aug[i], self.weights))

        y_pred = np.where(scores >= 0.0, 1, -1)

        if input_was_1d:
            return int(y_pred[0])
        else:
            return y_pred
        
    def train(self, training_inputs, labels):
        """
        Trains the perceptron.
        Args:
            training_inputs (list of np.ndarray): List of numpy arrays of training points.
            labels (np.ndarray): Array of expected output values for the corresponding point in training_inputs.
        """
        X = np.asarray(training_inputs, dtype=float)   # (n, d)
        y = np.asarray(labels)

        ones = np.ones((X.shape[0], 1), dtype=float)
        X_aug = np.concatenate([X, ones], axis=1)     # (n, d+1)

        # we need max_epochs to train our model
        for epoch in range(self.max_epochs): 
            """
                What we should do in one epoch ? 
                you are required to write code for 
                1.do forward pass
                2.calculate the error
                3.compute parameters' gradient 
                4.Using gradient descent method to update parameters(not Stochastic gradient descent!,
                please follow the algorithm procedure in "perceptron_tutorial.pdf".)
            """
        
            scores = np.zeros(X_aug.shape[0], dtype=float)
            for i in range(X_aug.shape[0]):
                scores[i] = float(np.dot(X_aug[i], self.weights))
            y_pred = np.where(scores >= 0.0, 1, -1)

            mis_indices = []
            for i in range(X_aug.shape[0]):
                if y_pred[i] != y[i]:
                    mis_indices.append(i)

            grad = np.zeros_like(self.weights, dtype=float)
            for i in mis_indices:
                grad = grad + y[i] * X_aug[i]

            self.weights = self.weights + self.learning_rate * grad

            # if epoch%50==49:
            #     print(f"Epoch {epoch+1}: misclassified = {len(mis_indices)}")
