import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Initialize weights and biases
def init():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    w3 = np.random.rand(10, 10) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2, w3, b3

# Functions
def ReLU(z):
    return np.maximum(0.01 * z, z)

def Softmax(z):
    z -= np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def OneHot(y):
    onehot = np.zeros((y.size, y.max() + 1))
    onehot[np.arange(y.size), y] = 1
    return onehot.T

# Forward and backward propagation
def forward(x, w1, b1, w2, b2, w3, b3):
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = ReLU(z2)
    z3 = w3.dot(a2) + b3
    a3 = Softmax(z3)
    return z1, a1, z2, a2, z3, a3

def back(x, y, z1, a1, z2, a2, z3, a3, w1, w2, w3):
    one_hot_y = OneHot(y)

    dz3 = a3 - one_hot_y
    dw3 = 1 / y.size * dz3.dot(a2.T)
    db3 = 1 / y.size * np.sum(dz3, axis=1, keepdims=True)

    dz2 = w3.T.dot(dz3) * (z2 > 0)
    dw2 = 1 / y.size * dz2.dot(a1.T)
    db2 = 1 / y.size * np.sum(dz2, axis=1, keepdims=True)

    dz1 = w2.T.dot(dz2) * (z1 > 0)
    dw1 = 1 / y.size * dz1.dot(x.T)
    db1 = 1 / y.size * np.sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2, dw3, db3

# Update weights and biases
def update(dw1, db1, dw2, db2, dw3, db3, w1, b1, w2, b2, w3, b3, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    w3 = w3 - alpha * dw3
    b3 = b3 - alpha * db3
    return w1, b1, w2, b2, w3, b3

def predict(a3):
    return np.argmax(a3, axis=0)

def accuracy(predictions, y):
    return np.sum(predictions == y) / y.size

def gradient_descent(x, y, iterations, alpha):
    w1, b1, w2, b2, w3, b3 = init()
    for i in range(iterations):
        z1, a1, z2, a2, z3, a3 = forward(x, w1, b1, w2, b2, w3, b3)
        dw1, db1, dw2, db2, dw3, db3 = back(x, y, z1, a1, z2, a2, z3, a3, w1, w2, w3)
        w1, b1, w2, b2, w3, b3 = update(dw1, db1, dw2, db2, dw3, db3, w1, b1, w2, b2, w3, b3, alpha)

        if i % 50 == 0:
            predictions = predict(a3)
            print("Iteration: ", i)
            print(accuracy(predictions, y))
    return w1, b1, w2, b2, w3, b3

# Make predictions
def make_predictions(x, w1, b1, w2, b2, w3, b3):
    _, _, _, _, _, a3 = forward(x, w1, b1, w2, b2, w3, b3)
    predictions = predict(a3)
    return predictions

def test_prediction(index, w1, b1, w2, b2, w3, b3):
    img = x_dev[:, index, None]
    x = x_dev[:, index, None]
    y = y_dev[index]
    prediction = make_predictions(x, w1, b1, w2, b2, w3, b3)
    print("Prediction: ", prediction)
    print("Actual: ", y)

    plt.gray()
    plt.imshow(img.reshape(28, 28), interpolation='nearest')
    plt.show()

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

    w1, b1, w2, b2, w3, b3 = gradient_descent(x_train, y_train, 1000, 0.1)

    test_predictions = make_predictions(x_dev, w1, b1, w2, b2, w3, b3)

    print("\nAccuracy on test data: ", accuracy(test_predictions, y_dev))

    out_df = pd.DataFrame({'Actual': y_dev, 'Prediction': test_predictions})
    out_df.to_csv('results.csv')