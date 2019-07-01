import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df_wine = pd.read_csv('/Users/chenshuran/Sites/pythonML/chapter5/wine.data', 
			header = None)
df_wine.columns = ['Class label', 'Alcohol',
		   'Malic acid', 'Ash',
		   'Alcalinity of ash', 'Magnesium',
		   'Total phenols', 'Flavanoids',
		   'Nonflavanoid phenols',
		   'Proanthocyanins',
		   'Color intensity', 'Hue',
		   'OD280/OD315 of diluted wines',
		   'Proline']

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
						    stratify=y,
						    random_state=0)

#standardize the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#obtain the eigenpairs of the covariance matrix
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)


tot = sum(eigen_vals)
var_exp = [(i/ tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
		label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid',
		label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()
