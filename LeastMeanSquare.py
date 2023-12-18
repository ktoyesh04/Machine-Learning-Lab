import numpy as np

N = 1000
LEARNING_RATE = 0.01

X = np.array([[0.2, 0.4], [0.4, 0.6], [0.6, 0.8]])
y_true = np.array([0, 1, 1])
y_true[0] = 1.5

X_bias = np.c_[np.ones((X.shape[0], 1)), X]
initial_weights = np.random.randn(X_bias.shape[1], 1)
weights = initial_weights.copy()

for _ in range(N):
    y_pred = X_bias.dot(weights).flatten()
    errors = y_true - y_pred
    gradients = X_bias.T.dot(errors.reshape(-1, 1))
    weights = weights + LEARNING_RATE * gradients

print("Initial Weights:\n",initial_weights)
print("Errors:", errors)
print("Updated Weights:\n", weights)
