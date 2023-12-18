import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt

data = pd.read_csv('data.csv')

dummy_data = pd.get_dummies(data, columns=data.columns[:-1])
X = dummy_data.drop('PlayTennis', axis=1)
y = dummy_data['PlayTennis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


dtree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, labels=['Yes', 'No'])

print('Accuracy:', accuracy)
print('Report:', report)

fig = plt.figure(figsize=(16, 9))
a = plot_tree(dtree, feature_names=list(X.columns), filled=True, fontsize=12, class_names=['No', 'Yes'])
