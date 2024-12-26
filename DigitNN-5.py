import numpy as np
import pandas as pd
from PIL import Image

# Activation functions
def ReLU(z):
    return np.maximum(0, z)

def ReLU_derivative(z):
    # Explicit float casting for clarity
    return (z > 0).astype(float)

def Tanh(z):
    return np.tanh(z)

def Tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

def Softmax(z):
    # Shift by max for numerical stability
    z -= np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# --------------------------------------------------

class Layer:
    """
    A single layer in a neural network.
    """
    def __init__(self, input_size, output_size, activation, activation_derivative):
        #self.weights = np.random.rand(output_size, input_size) - 0.5
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.biases  = np.zeros((output_size, 1))
        self.activation = activation
        self.activation_derivative = activation_derivative

        # Placeholders for forward/backward passes:
        self.z = None  # Weighted sums
        self.a = None  # Post-activation

        # Gradients:
        self.dz = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        Forward pass through this layer.
        :param x: Input data of shape (input_size, number_of_examples)
        :return: Post-activation values of shape (output_size, number_of_examples)
        """
        self.z = self.weights @ x + self.biases
        self.a = self.activation(self.z) if self.activation else self.z
        return self.a

    def back(self, x_prev, y, w_next, dz_next):
        """
        Backward pass: compute gradients for this layer.
        :param x_prev: Activations from the previous layer (or input X if first layer).
        :param y: One-hot labels of shape (num_classes, number_of_examples).
        :param w_next: Weights of the next layer (None if this is the last layer).
        :param dz_next: dZ from the next layer (None if this is the last layer).
        :return: dZ for this layer, to propagate backward further.
        """
        m = y.shape[1]  # number of examples

        # If this is the output layer:
        if w_next is None and dz_next is None:
            # For a Softmax + Cross Entropy, dZ = (prediction - one_hot_target)
            self.dz = self.a - y
        else:
            # Otherwise use chain rule: dZ = W_next^T * dZ_next * f'(Z)
            self.dz = (w_next.T @ dz_next) * self.activation_derivative(self.z)

        # dW = 1/m * dZ * (A_prev)^T
        self.dw = (1 / m) * self.dz @ x_prev.T
        # dB = 1/m * sum(dZ)
        self.db = (1 / m) * np.sum(self.dz, axis=1, keepdims=True)

        return self.dz

    def update(self, alpha):
        """
        Gradient descent parameter update.
        :param alpha: Learning rate
        """
        self.weights -= alpha * self.dw
        self.biases  -= alpha * self.db

# --------------------------------------------------

class NeuralNetwork:
    """
    A simple fully-connected feedforward neural network.
    """
    def __init__(self, layers):
        self.layers = layers

    @staticmethod
    def OneHot(y):
        """
        Convert integer labels into one-hot encoded vectors.
        """
        onehot = np.zeros((y.size, y.max() + 1))
        onehot[np.arange(y.size), y] = 1
        return onehot.T

    def forward(self, x):
        """
        Forward propagate through the entire network.
        :param x: Input data, shape (input_size, number_of_examples)
        :return: The final output layer's activation
        """
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def back(self, x, y_onehot):
        """
        Backward propagate through the entire network.
        :param x: Original input data
        :param y_onehot: One-hot encoded labels
        """
        # Go in reverse order of layers
        for i in reversed(range(len(self.layers))):
            if i == len(self.layers) - 1:
                # Output layer: no w_next or dz_next
                self.layers[i].back(self.layers[i-1].a, y_onehot, None, None)
            elif i == 0:
                # First layer: x is the input
                self.layers[i].back(x, y_onehot, self.layers[i+1].weights, self.layers[i+1].dz)
            else:
                # Hidden layers
                self.layers[i].back(self.layers[i-1].a, y_onehot, self.layers[i+1].weights, self.layers[i+1].dz)

    def update(self, alpha):
        """
        Update the weights and biases of all layers.
        """
        for layer in self.layers:
            layer.update(alpha)

    def predict(self, x):
        """
        Make predictions with the trained model.
        :param x: Input data, shape (input_size, number_of_examples)
        :return: Predicted integer class labels
        """
        output = self.forward(x)
        return np.argmax(output, axis=0)

    @staticmethod
    def accuracy(y, y_pred):
        """
        Compute the accuracy.
        :param y: True integer labels
        :param y_pred: Predicted integer labels
        :return: Float accuracy value
        """
        return np.mean(y == y_pred)

    def train(self, x, y, iterations, alpha, print_every=50):
        """
        Train the neural network with full-batch gradient descent.
        :param x: Training data
        :param y: Integer labels for training data
        :param iterations: Number of gradient descent steps
        :param alpha: Learning rate
        :param print_every: Frequency of printing accuracy
        """
        # Convert y into one-hot outside the main loop for efficiency
        y_onehot = self.OneHot(y)

        for i in range(iterations):
            # Decay learning rate every 20 iterations
            alpha = alpha * (0.99 ** (i // 100))
            # Forward pass
            self.forward(x)
            # Backward pass
            self.back(x, y_onehot)
            # Update parameters
            self.update(alpha)

            # Print accuracy every 'print_every' iterations
            if i % print_every == 0:
                predictions = self.predict(x)
                acc = self.accuracy(y, predictions)
                print(f"Iteration {i}: accuracy = {acc:.4f}")

# --------------------------------------------------

if __name__ == '__main__':
    # 1. Load and shuffle data
    df = pd.read_csv('train.csv')
    data = np.array(df)
    np.random.shuffle(data)

    # 2. Split into dev and train sets
    data_dev  = data[:1000].T
    y_dev     = data_dev[0]
    x_dev     = data_dev[1:] / 255.0

    data_train = data[1000:].T
    y_train    = data_train[0]
    x_train    = data_train[1:] / 255.0

    # 3. Create a network with 3 layers
    layers = [
        Layer(784,   16,   ReLU,        ReLU_derivative),
        Layer(16,  16,   ReLU,        ReLU_derivative),
        Layer(16,  16,   ReLU,        ReLU_derivative),
        Layer(16,  10,     Softmax,     None)
    ]
    nn = NeuralNetwork(layers)

    # 4. Train the network
    nn.train(x_train, y_train, iterations=1000, alpha=0.075, print_every=50)

    # 5. Test on the dev set
    test_predictions = nn.predict(x_dev)
    test_accuracy = nn.accuracy(y_dev, test_predictions)
    print(f"\nAccuracy on test data: {test_accuracy:.4f}")

    # 6. Save predictions to CSV
    out_df = pd.DataFrame({'Actual': y_dev, 'Prediction': test_predictions})
    out_df.to_csv('results.csv', index=False)
    print("Results saved to 'results.csv'.\n")

    # 7. Enter file names to test
    while True:
        file_name = input("Enter a file name to test (or 'q' to quit): ")
        if file_name == 'q':
            break
        img = Image.open(file_name).convert('L')
        img = np.array(img).reshape(784, 1) / 255.0
        prediction = nn.predict(img)
        print(f"Prediction: {prediction[0]}")