import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import sklearn
from random import shuffle
from sklearn.datasets import fetch_california_housing
import copy

def AdaBoostProcess():
    # Create stumps
    # bin the data by proportion, 10% in each bin
    bins = {}
    bin_idx = (np.arange(0, 1.1, 0.1)*num_samples).astype(np.int16)
    bin_idx[-1] = bin_idx[-1]-1
    for feature in (feature_names):
        bins[feature] = np.sort(data[feature])[bin_idx]

    # decision stumps as weak classifiers
    # 0 if not in bin, 1 if in bin
    stumps = {}
    for feature in feature_names:
        stumps[feature] = np.zeros([num_samples, len(bins[feature])-1])
        for n in range(len(bins[feature])-1):
            stumps[feature][:, n] = data[feature] > bins[feature][n]

    # stack the weak classifiers into a matrix
    H = np.hstack([stumps[feature] for feature in feature_names])
    H = np.hstack([np.ones([num_samples, 1]), H])
    # prepare the vector for storing weights
    alphas = np.zeros(H.shape[1])
    # Apply AdaBoost
    num_iterations = 30
    MSE = np.zeros(num_iterations) # track mean square error

    for iteration in range(num_iterations):
        f = H.dot(alphas)  # the current f(x) - linear combination of weak classifiers
        r = y-f
        MSE[iteration] = np.mean(r**2)  # r = residual
        loss_sums = np.matmul(r.T, H)
        idx = np.argmax(abs(loss_sums))  # optimal direction to move in
        count_ones_idx = sum(H.T[idx])
        update_weight = loss_sums[idx] / count_ones_idx
        alphas[idx] = alphas[idx] + update_weight
        # amount to move in optimal direction

    return bins, stumps, H, alphas

def PlotResults(filename = "foo.png"):
    plt.rcParams['font.size'] = 10
    alphasf = {}
    start = 1
    for feature in feature_names:
        alphasf[feature] = alphas[start:(start + stumps[feature].shape[1])]
        start = start + stumps[feature].shape[1]
    alphasf['mean'] = alphas[0]
    fig = plt.figure()
    fig.subplots_adjust(hspace=1.5)
    plt.suptitle("Contribution of feature to house price")
    i = 1
    for feature in feature_names:
        ax = fig.add_subplot(330 + i)
        ax.plot(data[feature], y - np.mean(y), '.', alpha=0.2, color=[0.9, 0.9, 0.9], ms=0.5)
        ax.plot(data[feature], stumps[feature].dot(alphasf[feature]) - np.mean(stumps[feature].dot(alphasf[feature])),
                '.', alpha=0.2, color='b', ms=0.2)
        # plot stuff
        ax.title.set_text(feature)
        ax.set_xlim([bins[feature][0], bins[feature][-2]])
        ax.tick_params(labelsize=8)
        ax.set_yticks(np.arange(-2, 3, 1))
        i += 1
    #plt.show()
    plt.savefig(filename, bbox_inches='tight')

# Download data
tmp = sklearn.datasets.fetch_california_housing()

num_samples = tmp['data'].shape[0]
feature_names = tmp['feature_names']
y = tmp['target']
X = tmp['data']

data = {}
for n, feature in enumerate(feature_names):
    data[feature] = tmp['data'][:, n]

fig = plt.figure()
fig.subplots_adjust(hspace=1.5)
plt.suptitle("House Price vs Feature")
i = 1
for feature in feature_names:
    ax = fig.add_subplot(330 + i)
    ax.plot(data[feature], y - np.mean(y), '.', alpha=0.5, color='b', ms=0.5)
    # plot stuff
    ax.title.set_text(feature)
    ax.tick_params(labelsize=8)
    ax.set_yticks(np.arange(-2, 3, 1))
    i += 1
#plt.show()
plt.savefig("first_plot.png", bbox_inches='tight')

#Original Data
bins, stumps, H, alphas = AdaBoostProcess()
MSE_original = np.mean((y - H.dot(alphas))**2)
# Plot Results
PlotResults(filename="original_data_plot.png")

#Scramble Data to calculate variable importance
original_data = copy.deepcopy(data)
for feature in feature_names:
    np.random.shuffle(data[feature])
    bins, stumps, H, alphas = AdaBoostProcess()
    MSE_new = np.mean((y - H.dot(alphas)) ** 2)
    print("Variable Importance of feature " + str(feature) + ": " + str(round(MSE_new - MSE_original,3)))
    PlotResults(filename=str(feature) + "_plot.png")
    data = copy.deepcopy(original_data)