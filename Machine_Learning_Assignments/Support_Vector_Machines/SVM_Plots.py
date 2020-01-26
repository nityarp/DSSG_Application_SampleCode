import numpy as np
import csv
import math
from sklearn import svm, preprocessing
from sklearn.metrics import zero_one_loss
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

def read_data(file):
    data = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # take the header out
        for row in reader:  # each row is a list
            data.append(row)
    data = np.array(data, dtype=float)
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def SVMPlot(kernel, gamma, C):
    clf = svm.SVC(kernel="rbf", C=C, gamma=gamma)
    clf.fit(X_train, y_train)
    train_predicted = clf.predict(X_train)

    # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(1, figsize=(6, 4))
    plt.clf()

    plt.axis('tight')
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(1, figsize=(4, 3))
    plt.contourf(XX, YY, Z, levels=[-1.3, -1, -0.5, 0, 0.5, 1, 1.3], cmap=cm.RdBu)

    y_labels = []
    num_vals = len(X_train) - len(clf.support_)
    X_train_svremoved = np.zeros(shape=[num_vals, 2])
    y_train_svremoved = np.zeros(shape=[num_vals])

    j = 0
    for i, xi in enumerate(X_train):
        if i not in clf.support_:
            X_train_svremoved[j] = [xi[0], xi[1]]
            y_train_svremoved[j] = y_train[i]
            j += 1

    for index in clf.support_:
        y_labels.append(y_train[index])

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], c=y_labels, cmap=cm.RdBu, edgecolors='k', s=20,
                alpha=0.3)
    plt.scatter(X_train_svremoved[:, 0], X_train_svremoved[:, 1], c=y_train_svremoved, cmap=plt.cm.RdBu, s=20,
                edgecolors="k")
    plt.contour(XX, YY, Z, levels=[-1, 0, 1], linestyles=["--"], color=[0,0,0])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    plt.show()
    # plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');

train_file = 'svm_dataset/train.csv'
X_train, y_train = read_data(train_file)

test_file = 'svm_dataset/test.csv'
X_test, y_test = read_data(test_file)

std_scale = preprocessing.StandardScaler().fit(X_train)
X_train = std_scale.transform(X_train)
X_test = std_scale.transform(X_test)

#2c - Support Vector Machine Plot
gammas = [1, 10, 100]
for gamma_val in gammas:
    C = 1
    SVMPlot("rbf", gamma_val, C)

Cs = [1, 10, 100]
for c_val in Cs:
    gamma = 1
    SVMPlot("rbf", gamma, c_val)