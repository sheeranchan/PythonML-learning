from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print 'Class labels:',np.unique(y)

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size = 0.3, random_state = 1, stratify = y)

"""
# test sklearn shuffle using Numpy bincount to count their occurances
print 'Labels counts in y: ', np.bincount(y)
print 'Labels counts in y_train: ', np.bincount(y_train)
print 'Labels counts in y_test: ', np.bincount(y_test)
"""

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# modeling
ppn = Perceptron(n_iter = 40, eta0 = 0.01, random_state = 1)
ppn.fit(X_train_std, y_train)

# prediction
y_pred = ppn.predict(X_test_std)
print 'Misclassified samples: ', (y_test != y_pred).sum()

# check classification accuracy of the perceptron on the test set
print 'Accuracy in test set:', accuracy_score(y_test, y_pred)
print 'Accuracy by score method in standardized test set:',ppn.score(X_test_std, y_test)

