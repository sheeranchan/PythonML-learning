import matplotlib.pyplot as plt
import plotDecisionRegions as pdr
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import LogisticRegressionGD as lrGD

# get training dataset from sklearn library
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# seperate training & testing data sets
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size = 0.3, random_state = 1, stratify = y)

X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
lrgd = lrGD.LogisticRegressionGD(eta = 0.05,
				n_iter = 1000,
				random_state = 1)
lrgd.fit(X_train_01_subset, y_train_01_subset)


pdr.plot_decision_regions(X = X_train_01_subset,
			y = y_train_01_subset, 
			classifier = lrgd)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc= 'upper left')
plt.show()

