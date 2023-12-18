import numpy as np


def activate(z: int) -> int:
    return 1 if z > 0 else 0


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0, 0, 1])
weights = np.array([0.1, 0.2])
bias, learn = 0.3, 0.1
epochs = 100

for _ in range(epochs):
    for x, y in zip(X, Y):
        z = np.dot(x, weights) + bias
        prediction = activate(z)
        weights += learn * (y - prediction) * x
        bias += learn * (y - prediction)

tests = X
for test in tests:
    z = np.dot(test, weights) + bias
    prediction = activate(z)
    print(f"input: {test}, prediction: {prediction}")
