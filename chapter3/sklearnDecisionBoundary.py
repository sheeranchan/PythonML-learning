import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

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
ppn = Perceptron(n_iter = 40, eta0 = 0.01, random_state = 1)
ppn.fit(X_train_std, y_train)

# plot regions
def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.02):

	# setup marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	# plot the decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
				np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	#plot class samples
	for idx, cl in enumerate(np.unique(y)):
	    plt.scatter(x = X[y == cl, 0],
			y = X[y == cl, 1],
			alpha = 0.8,
			c = colors[idx],
			marker = markers[idx],
			label = cl,
			edgecolor = 'black')
	#highlight test samples
	if test_idx:
	    # plot all samples
	    X_test, y_test = X[test_idx, :], y[test_idx]

	    plt.scatter(X_test[:, 0], X_test[:, 1],
			c = '', edgecolor = 'black', alpha = 1.0,
			s = 100, label = 'test set')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X = X_combined_std,
			y = y_combined, 
			classifier = ppn,
			test_idx = range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc= 'upper left')
plt.show()

