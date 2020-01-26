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


def kernels(xi, xj, kernel, degree=0):
    if kernel == "rbf":
        l2_norm = np.linalg.norm(xi - xj)
        return math.exp(-gamma * (l2_norm ** 2))  # rbf-kernel calculation
    if kernel == "linear":
        return xi.dot(xj)
    if kernel == "poly":
        return (1 + xi.dot(xj) / C) ** degree


def lssvm_model(X, y, kernel, degree=0):
    obs_count = len(X)
    # to solve the linear system for LS-SVM
    equation_coefficients = np.ones(shape=(obs_count + 1, obs_count + 1))
    equation_coefficients[0][0] = 0

    # Forming the omega matrix (nxn)
    for i, xi in enumerate(X):
        for j, xj in enumerate(X):
            # Calculating kernel to store in omega
            equation_coefficients[i + 1][j + 1] = kernels(xi, xj, kernel, degree)
            if i == j:
                equation_coefficients[i + 1][j + 1] += 1 / C

    # Creating the linear system to solve for alpha and b
    labels = [0] + [yi for yi in y]
    solution = np.linalg.solve(equation_coefficients, labels)
    return solution[0], solution[1:]


train_file = 'svm_dataset/train.csv'
X_train, y_train = read_data(train_file)

test_file = 'svm_dataset/test.csv'
X_test, y_test = read_data(test_file)

std_scale = preprocessing.StandardScaler().fit(X_train)
X_train = std_scale.transform(X_train)
X_test = std_scale.transform(X_test)

def lssvm_process(C, gamma, kernel, degree=0):
    b, alphas = lssvm_model(X_train, y_train, kernel, degree)
    # Calculate errors in training data
    # train_errors = [alpha / C for alpha in alphas]
    train_errors = np.zeros(shape=len(y_train))
    for i, x1 in enumerate(X_train):
        fx = y_train[i] - (alphas[i] / C)
        margin = y_train[i] * fx
        train_errors[i] = np.sign(margin)

    # Evaluate on test data
    obs_count = len(y_test)
    test_errors = np.zeros(shape=obs_count)
    for i, xi in enumerate(X_test):
        # Get sum of wT * phi(xi) + b - iterate over all training data
        fx = 0
        for j, xj in enumerate(X_train):
            fx += kernels(xi, xj, kernel, degree) * alphas[j]  # kernel calculation
        fx += b
        margin = y_test[i] * fx
        # test_errors[i] = 1 - margin
        test_errors[i] = np.sign(margin)

    # Take l2-norm to get average error
    # train_classification_error = np.linalg.norm([train_errors]) / len(train_errors)
    # test_classification_error = np.linalg.norm([test_errors]) / len(test_errors)

    # zero - one loss for classification error - percentage
    train_classification_error = sum(1 if sample < 0 else 0 for sample in train_errors) / len(train_errors)
    test_classification_error = sum(1 if sample < 0 else 0 for sample in test_errors) / len(test_errors)

    return train_classification_error, test_classification_error


def standardSVM_process(C, gamma, kernel, degree=0):
    # SVM Algorithm from sk-learn
    standard_svm = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    standard_svm.fit(X_train, y_train)
    train_predicted = standard_svm.predict(X_train)
    train_classification_error_stdsvm = zero_one_loss(y_train, train_predicted, normalize=True)
    test_predicted = standard_svm.predict(X_test)
    test_classification_error_stdsvm = zero_one_loss(y_test, test_predicted, normalize=True)

    return train_classification_error_stdsvm, test_classification_error_stdsvm


# USER DEFINED PARAMETERS
# Changing values to make box plots
C_values = [10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4]
gamma_values = [10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4]
kernel_values = ["linear", "rbf"]
degree = 3

lssvm_gamma_train_set = dict()
lssvm_gamma_test_set = dict()
lssvm_C_train_set = dict()
lssvm_C_test_set = dict()

stdsvm_gamma_train_set = dict()
stdsvm_gamma_test_set = dict()
stdsvm_C_train_set = dict()
stdsvm_C_test_set = dict()

for kernel in kernel_values:
    lssvm_C_train = list()
    lssvm_C_test = list()
    stdstv_C_train = list()
    stdstv_C_test = list()
    #Keep gamma constant
    for C in C_values:
        gamma = 1
        ls_svm_train_error, ls_svm_test_error = lssvm_process(C, gamma, kernel, degree)
        std_svm_train_error, std_svm_test_error = standardSVM_process(C, gamma, kernel, degree)

        lssvm_C_train.append(round(ls_svm_train_error,2))
        lssvm_C_test.append(round(ls_svm_test_error,2))
        stdstv_C_train.append(round(std_svm_train_error,2))
        stdstv_C_test.append(round(std_svm_test_error,2))

    #Keep C constant
    lssvm_gamma_train = list()
    lssvm_gamma_test = list()
    stdstv_gamma_train = list()
    stdstv_gamma_test = list()
    for gamma in gamma_values:
        C = 1
        ls_svm_train_error, ls_svm_test_error = lssvm_process(C, gamma, kernel, degree)
        std_svm_train_error, std_svm_test_error = standardSVM_process(C, gamma, kernel, degree)

        lssvm_gamma_train.append(round(ls_svm_train_error,2))
        lssvm_gamma_test.append(round(ls_svm_test_error,2))
        stdstv_gamma_train.append(round(std_svm_train_error,2))
        stdstv_gamma_test.append(round(std_svm_test_error,2))

    lssvm_gamma_train_set[kernel] = lssvm_gamma_train
    lssvm_gamma_test_set[kernel] = lssvm_gamma_test
    lssvm_C_train_set[kernel] = lssvm_C_train
    lssvm_C_test_set[kernel] = lssvm_C_test

    stdsvm_gamma_train_set[kernel] = stdstv_gamma_train
    stdsvm_gamma_test_set[kernel] = stdstv_gamma_test
    stdsvm_C_train_set[kernel] = stdstv_C_train
    stdsvm_C_test_set[kernel] = stdstv_C_test

print("LSSVM")
print("Train Errors - Varying Gamma, C = 1")
print(lssvm_gamma_train_set)
print("Test Errors - Varying Gamma, C = 1")
print(lssvm_gamma_test_set)

print("Train Errors - Varying C, Gamma = 1")
print(lssvm_C_train_set)
print("Test Errors - Varying C, Gamma = 1")
print(lssvm_C_test_set)

print("Standard SVM")
print("Train Errors - Varying Gamma, C = 1")
print(stdsvm_gamma_train_set)
print("Test Errors - Varying Gamma, C = 1")
print(stdsvm_gamma_test_set)

print("Train Errors - Varying C, Gamma = 1")
print(stdsvm_C_train_set)
print("Test Errors - Varying C, Gamma = 1")
print(stdsvm_C_test_set)

#LSSVM plots by varying C
lssvm_boxplot_fig, lssvm_ax = plt.subplots()
data = [lssvm_C_train_set['rbf'], lssvm_C_test_set['rbf']]
        #lssvm_C_train_set['linear'], lssvm_C_test_set['linear']]
        #lssvm_C_train_set['poly'], lssvm_C_test_set['poly']]
lssvm_ax.set_title('LS-SVM Classification Error Rate - Varying C')
lssvm_ax.boxplot(data)
plt.xticks([1, 2], ['RBF - train', 'RBF - test'])
plt.show()

#LSSVM plots by varying gamma
lssvm_boxplot_fig, lssvm_ax = plt.subplots()
data = [lssvm_gamma_train_set['rbf'], lssvm_gamma_test_set['rbf']]
        #lssvm_gamma_train_set['linear'], lssvm_gamma_test_set['linear']]
        #lssvm_C_train_set['poly'], lssvm_C_test_set['poly']]
lssvm_ax.set_title('LS-SVM Classification Error Rate - Varying Gamma')
lssvm_ax.boxplot(data)
plt.xticks([1, 2], ['RBF - train', 'RBF - test'])
plt.show()

#Standard SVM plots by varying C
stdsvm_boxplot_fig, stdsvm_ax = plt.subplots()
data = [stdsvm_C_train_set['rbf'], stdsvm_C_test_set['rbf'],
        stdsvm_C_train_set['linear'], stdsvm_C_test_set['linear']]
        #lssvm_C_train_set['poly'], lssvm_C_test_set['poly']]
stdsvm_ax.set_title('Standard SVM Classification Error Rate - Varying C')
stdsvm_ax.boxplot(data)
plt.xticks([1, 2, 3, 4], ['RBF - train', 'RBF - test', 'Linear - train', 'Linear - test'])
plt.show()

#Standard plots by varying gamma
stdsvm_boxplot_fig, stdsvm_ax = plt.subplots()
data = [stdsvm_gamma_train_set['rbf'], stdsvm_gamma_test_set['rbf'],
        stdsvm_gamma_train_set['linear'], stdsvm_gamma_test_set['linear']]
        #lssvm_C_train_set['poly'], lssvm_C_test_set['poly']]
stdsvm_ax.set_title('Standard SVM Classification Error Rate - Varying Gamma')
stdsvm_ax.boxplot(data)
plt.xticks([1, 2, 3, 4], ['RBF - train', 'RBF - test', 'Linear - train', 'Linear - test'])
plt.show()
