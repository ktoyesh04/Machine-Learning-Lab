import numpy as np
import matplotlib.pyplot as plt


def sigmoid(sop) -> float:
    return 1.0 / (1 + np.exp(-sop))


def error(predicted, TARGET):
    return np.power(predicted - TARGET, 2)


def error_predicted_deriv(predicted, TARGET):
    return 2 * (predicted-TARGET)


def sigmoid_sop_deriv(sop):
    return sigmoid(sop) * (1.0 - sigmoid(sop))


def sop_w_deriv(x):
    return x


def update_w(w, grad):
    return w - LEARNING_RATE * grad


def plot_figure(y, ylabel):
    plt.figure()
    plt.plot(y)
    plt.title(f"Iteration Number vs {ylabel}")
    plt.xlabel("Iteration Number")
    plt.ylabel(ylabel)
    plt.show()


X1, X2 = 0.1, 0.4
TARGET, LEARNING_RATE = 0.7, 0.01
w1, w2 = np.random.rand(), np.random.rand()
print("Initial W : ", w1, w2)

predicted_output, network_error = [], []

for _ in range(80000):
    y = w1 * X1 + w2 * X2

    predicted = sigmoid(y)
    err = error(predicted, TARGET)
    predicted_output.append(predicted)
    network_error.append(err)

    g1 = error_predicted_deriv(predicted, TARGET)
    g2 = sigmoid_sop_deriv(y)

    g3w1 = sop_w_deriv(X1)
    g3w2 = sop_w_deriv(X2)

    gradw1 = g3w1 * g2 * g1
    gradw2 = g3w2 * g2 * g1

    w1 = update_w(w1, gradw1)
    w2 = update_w(w2, gradw2)

plot_figure(network_error, 'Error')
plot_figure(predicted_output, 'Prediction')
