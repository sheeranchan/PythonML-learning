import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from SequentialBackwardSelection import SBS
import pandas as pd
import numpy as np

df_wine = pd.read_csv('/Users/chenshuran/Sites/pythonML/chapter4/wine.data', 
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

#split the train & test dataset
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
						test_size=0.3,
						random_state=0,
						stratify=y)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)


#using KNN classifier to test the SBS method
knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)


#plot the diagram
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

