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

#compute d-dimensional mean vectors
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1,4):
	mean_vecs.append(np.mean(
		X_train_std[y_train==label], axis=0))
#	print('MV %s: %s\n' %(label, mean_vecs[label-1]))

#compute the within-class scatter matrices
d = 13 #number of features
S_W = np.zeros((d,d))
for label, mv in zip(range(1,4), mean_vecs):
	class_scatter = np.zeros((d,d))
	for row in X_train_std[y_train == label]:
		row, mv = row.reshape(d, 1), mv.reshape(d, 1)
		class_scatter += (row - mv).dot((row - mv).T)
	S_W += class_scatter

#print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
#print('Class label distribution: %s' % np.bincount(y_train)[1:])


#compute the between-class scatter matrices
mean_overall = np.mean(X_train_std, axis=0)
d = 13 #number of features
S_B = np.zeros((d,d))
for i, mean_vec in enumerate(mean_vecs):
        n = X_train[y_train == i + 1, :].shape[0]
	mean_vec = mean_vec.reshape(d, 1) # make column vector
	mean_overall = mean_overall.reshape(d, 1)
	S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
# print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))


# descending order of the eigenvalues
eigen_vals, eigen_vecs = \
		np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) 
		for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
#print('Eigenvalues in descending order:\n')
#for eigen_val in eigen_pairs:
#	print(eigen_val[0])

# display how much of discriminability information is captured
# by the linear discriminants (eigenvectors)
#tot = sum(eigen_vals.real)
#discr = [(i/ tot) for i in sorted(eigen_vals.real, reverse=True)]
#cum_discr = np.cumsum(discr)
#plt.bar(range(1, 14), discr, alpha=0.5, align='center',
#		label='individual "discriminability"')
#plt.step(range(1, 14), cum_discr, where='mid',
#		label='cumulative "discriminability"')
#plt.ylabel('"discriminability" ratio')
#plt.xlabel('Linear Discriminants')
#plt.ylim([-0.1, 1.1])
#plt.legend(loc='best')
#plt.show()


# create transformation matrix W
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
		eigen_pairs[1][1][:, np.newaxis].real))
#print('Matrix W: \n')
#print(w)

#project samples onto the new featuer space
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip (np.unique(y_train), colors, markers):
	plt.scatter(X_train_lda[y_train==l, 0],
		    X_train_lda[y_train==l, 1] * (-1),
		    c=c, label=l, marker=m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc = 'lower right')
plt.show()
