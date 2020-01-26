import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import random

def kmeans(X, K=5, maxiter=100):
    # initialize cluster centers
    C = np.array(random.choices(X, k=K))
    print(C)
    cluster_assignments = dict()

    for iter in range(maxiter):
        # cluster assignment update
        for x in X:
            closest_cluster = np.argmin(np.asarray([np.linalg.norm(x - c, ord=2)
                                                    for c in C]))
            if not closest_cluster in cluster_assignments.keys():
                cluster_assignments[closest_cluster] = list()
            cluster_assignments[closest_cluster].append(tuple(x))

        for k in range(K):
            # cluster center update
            C[k] = np.array(np.mean(cluster_assignments[k], axis=0))

    print('K-Means')
    print(C)
    return C

def kmedians(X, K=5, maxiter=100):
    # initialize cluster centers
    C = np.array(random.choices(X, k=K))
    cluster_assignments = dict()

    for iter in range(maxiter):
        # cluster assignment update
        for x in X:
            closest_cluster = np.argmin(np.asarray([np.linalg.norm(x - c, ord=1)
                                                    for c in C]))
            if not closest_cluster in cluster_assignments.keys():
                cluster_assignments[closest_cluster] = list()
            cluster_assignments[closest_cluster].append(tuple(x))

        for k in range(K):
            # cluster center update
            C[k] = np.array(np.median(cluster_assignments[k], axis=0))

    print('K-Medians')
    print(C)
    return C

tmp = sio.loadmat("mousetracks.mat")

tracks = {}
for trackno in range(30):
    tracks[trackno] = tmp["num%d" % (trackno)]

plt.close("all")
for trackno in range(30):
    plt.plot(tracks[(trackno)][:, 0], tracks[(trackno)][:, 1], '.')
plt.axis("square")
plt.xlabel("meters")
plt.ylabel("meters")
plt.show()

X = np.zeros([30 * 50, 2])

for trackno in range(30):
    X[(trackno * 50):((trackno + 1) * 50), :] = tracks[trackno]

#K Means
plt.close("all")
plt.plot(X[:, 0], X[:, 1], '.')
C = kmeans(X)
# uncomment to plot your cluster centers
plt.plot(C[:,0],C[:,1],'ro')
plt.axis("square")
plt.xlabel("meters")
plt.ylabel("meters")
plt.show()

#K Medians
plt.close("all")
plt.plot(X[:, 0], X[:, 1], '.')
C = kmedians(X)
# uncomment to plot your cluster centers
plt.plot(C[:,0],C[:,1],'ro')
plt.axis("square")
plt.xlabel("meters")
plt.ylabel("meters")
plt.show()