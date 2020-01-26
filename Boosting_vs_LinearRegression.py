import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import sklearn
from random import shuffle
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.linear_model import LinearRegression

# Download data
tmp = fetch_california_housing()

num_samples = tmp['data'].shape[0]
feature_names = tmp['feature_names']
y = tmp['target']
X = tmp['data']

clf = GradientBoostingRegressor(loss="ls")
clf.fit(X,y)

plt.close("all")
plt.figure(figsize=[10,10])
ax = plt.gca()
plot_partial_dependence(clf, X, feature_names, feature_names, n_cols=3, ax=ax)
plt.tight_layout()
plt.show()

clf2 = LinearRegression()
clf2.fit(X,y)

MSE_boosting = np.mean((y-clf.predict(X))**2)
MSE_LR = np.mean((y-clf2.predict(X))**2)

print("MSE of Boosting = " + str(MSE_boosting))
print("MSE of LR = " + str(MSE_LR))