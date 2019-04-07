from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
from sklearn.tree import DecisionTreeClassifier
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
tree = DecisionTreeClassifier(criterion = 'gini',
				max_depth = 4, 
				random_state = 1)
tree.fit(X_train_std, y_train)

dot_data = export_graphviz(tree,
			   filled = True,
			   rounded = True,
			   class_names = ['Setosa',
					'Versicolor',
					'Virginica'],
			   feature_names = ['petal length',
					'petal width'],
			   out_file = None)
graph = graph_from_dot_data(dot_data)
graph.write_png('/Users/chenshuran/Documents/PythonML/chapter3/tree.png')


