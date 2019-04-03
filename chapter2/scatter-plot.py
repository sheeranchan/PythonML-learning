import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# get training dataset
df = pd.read_csv('/Users/chenshuran/Documents/PythonML/datasets/iris_dataset.csv', header = None)

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0,2]].values

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
		color = 'red', marker='o', label = 'setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
		color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc= 'upper left')
plt.show()

