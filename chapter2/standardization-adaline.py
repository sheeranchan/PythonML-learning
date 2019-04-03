import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap 
import numpy as np
import pandas as pd
import adalineGD as ada

# get training dataset
df = pd.read_csv('/Users/chenshuran/Documents/PythonML/datasets/iris_dataset.csv', header = None)

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0,2]].values

# standardization: give data the property of a standard normal distribution, which help gradient feature
# helping gradient descent learning to converge more quickly
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()


# plot data
ada = ada.AdalineGD(n_iter = 15, eta = 0.01)
ada.fit(X_std, y)

def plot_decision_regions(X, y, classifier, resolution = 0.02):

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
	    plt.scatter(x = X[ y == cl, 0],
			y = X[ y == cl, 1],
			alpha = 0.8,
			c = colors[idx],
			marker = markers[idx],
			label = cl,
			edgecolor = 'black')

plot_decision_regions(X_std, y, classifier = ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()


plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()
