import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df_wine = pd.read_csv('/Users/chenshuran/Sites/pythonML/chapter5/wine.data', 
			header = None)
#df_wine.columns = ['Class label', 'Alcohol',
#		   'Malic acid', 'Ash',
#		   'Alcalinity of ash', 'Magnesium',
#		   'Total phenols', 'Flavanoids',
#		   'Nonflavanoid phenols',
#		   'Proanthocyanins',
#		   'Color intensity', 'Hue',
#		   'OD280/OD315 of diluted wines',
#		   'Proline']

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
						    stratify=y,
						    random_state=0)

#standardize the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Obtain the eigenpairs of the covariance matrix
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# Make a list of (eigenvalues, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) 
			for i in range (len(eigen_vals))];
# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# Create a 13X2-dimensional projection matrix W
W = np.hstack((eigen_pairs[0][1][:, np.newaxis],
		eigen_pairs[1][1][:, np.newaxis]))

# Transfer the entire 124X13-dimensional training dataset 
# onto the two principal components by calculationg the matrix dot product
X_train_pca = X_train_std.dot(W)


colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
	plt.scatter(X_train_pca[y_train==l, 0],
		    X_train_pca[y_train==l, 1],
		    c = c, label=l, marker=m)

plt.ylabel('PC 1')
plt.xlabel('PC 2')
plt.legend(loc='lower left')
plt.show()
