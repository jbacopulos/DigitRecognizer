import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Layer:
    def __init__(self, input_size, output_size, activation, activation_derivative):
        self.weights = np.random.rand(output_size, input_size) - 0.5
        self.biases = np.random.rand(output_size, 1) - 0.5
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, x):
        self.z = self.weights.dot(x) + self.biases
        self.a = self.activation(self.z)
        return self.a
    
    def back(self, x, y, w_next, dz_next):
        if w_next is None and dz_next is None:
            self.dz = self.a - y
        else:
            self.dz = w_next.T.dot(dz_next) * self.activation_derivative(self.z)
        self.dw = 1 / y.size * self.dz.dot(x.T)
        self.db = 1 / y.size * np.sum(self.dz, axis=1, keepdims=True)
        return self.dz
    
    def update(self, alpha):
        self.weights = self.weights - alpha * self.dw
        self.biases = self.biases - alpha * self.db

class NeuralNetwork:
    # Initialize weights and biases
    def __init__(self, layers):            
        self.layers = layers

    def OneHot(self, y):
        onehot = np.zeros((y.size, y.max() + 1))
        onehot[np.arange(y.size), y] = 1
        return onehot.T

    # Forwards and backwards propagation
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        
    def back(self, x, y):
        y = self.OneHot(y)
        dz = None

        for i in range(len(self.layers) - 1, -1, -1):
            if i == len(self.layers) - 1:
                dz = self.layers[i].back(self.layers[i - 1].a, y, None, None)
            elif i == 0:
                dz = self.layers[i].back(x, y, self.layers[i + 1].weights, dz)
            else:
                dz = self.layers[i].back(self.layers[i - 1].a, y, self.layers[i + 1].weights, dz)

    # Update weights and biases
    def update(self, alpha):
        for layer in self.layers:
            layer.update(alpha)

    def predict(self, x=None):
        if x is None:
            return np.argmax(self.layers[-1].a, axis=0)
        else:
            self.forward(x)
            return self.predict()

    def accuracy(self, y, predictions):
        return np.sum(predictions == y) / y.size

    def train(self, x, y, iterations, alpha):
        for i in range(iterations):
            self.forward(x)
            self.back(x, y)
            self.update(alpha)

            if i % 50 == 0:
                predictions = self.predict()
                print("Iteration: ", i)
                print(self.accuracy(y, predictions))
    
# Functions
def ReLU(z):
    return np.maximum(0, z)

def ReLU_derivative(z):
    return z > 0

def Softmax(z):
    z -= np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    data = np.array(df)
    np.random.shuffle(data)

    data_dev = data[:1000].T
    y_dev = data_dev[0]
    x_dev = data_dev[1:] / 255.0

    data_train = data[1000:].T
    y_train = data_train[0]
    x_train = data_train[1:] / 255.0

    layers = [
        Layer(784, 10, ReLU, ReLU_derivative),
        Layer(10, 10, ReLU, ReLU_derivative),
        Layer(10, 10, Softmax, None)
    ]

    nn = NeuralNetwork(layers)
    nn.train(x_train, y_train, 1500, 0.1)

    test_predictions = nn.predict(x_dev)

    print("\nAccuracy on test data: ", nn.accuracy(y_dev, test_predictions))

    out_df = pd.DataFrame({'Actual': y_dev, 'Prediction': test_predictions})
    out_df.to_csv('results.csv')