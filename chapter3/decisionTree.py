import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import plotDecisionRegions as pdr

# get training dataset from sklearn library
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# seperate training & testing data sets
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size = 0.3, random_state = 1, stratify = y)

# standardisation
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# modeling
tree = DecisionTreeClassifier(criterion = 'gini',
				max_depth = 4, 
				random_state = 1)
tree.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
pdr.plot_decision_regions(X = X_combined_std,
			y = y_combined, 
			classifier = tree,
			test_idx = range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc= 'upper left')
plt.show()

