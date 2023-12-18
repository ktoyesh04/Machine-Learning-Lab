import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

data = pd.read_csv('knn.csv')
X, y = data.iloc[:, -3:-1].values, data.iloc[:, -1].values

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=0)
stX = StandardScaler()
X_train1, X_test1 = stX.fit_transform(X_train), stX.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=2, metric='euclidean')
classifier.fit(X_train1, y_train)

y_pred = classifier.predict(X_test1)

print(confusion_matrix(y_test, y_pred))


# Plotting KNN (optional)
def plot_decision_boundary(x_set, y_set, classifier, title):
    x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                        np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))

    plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))

    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
        
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()


plot_decision_boundary(X_train1, y_train, classifier, 'K-NN Algorithm (Training set)')
plot_decision_boundary(X_test1, y_test, classifier, 'K-NN Algorithm (Test set)')
