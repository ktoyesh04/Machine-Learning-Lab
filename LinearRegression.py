import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('saless.csv')
X_train = df['X'].values.reshape(-1, 1)
y_train = df['y'].values
X_test = np.array([7, 12]).reshape(-1, 1) 

model = LinearRegression()
model.fit(X_train, y_train)

a = model.intercept_
b = model.coef_[0]
print(f"Estimated parameters: a = {a}, b = {b}")

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print(f"Predicted sales for the testing data: {y_pred_test}")

plt.title('Linear Regression Example with Custom Dataset')
plt.xlabel('X')
plt.ylabel('y')

plt.scatter(X_train, y_train, color='black', label='Actual Data')
plt.plot(X_train, y_pred_train, color='blue', linewidth=3, label='Linear Regression Line')
plt.scatter(X_test, y_pred_test, color='red', label='Predicted Data (Testing)')

plt.legend()
plt.show()
