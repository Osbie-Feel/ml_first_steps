# -*- coding: utf-8 -*-
"""
===================================
Demo of DBSCAN clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.

"""

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt


# #############################################################################
# Generate sample data
iris = datasets. load_iris()
X = iris.data
labels_true = iris.target
X = StandardScaler().fit_transform(X)
ldbs = []
leps = [0.5, 0.6, 0.7]
lsamples = [5, 10, 15]

for eps in leps:
    for min_samples in lsamples:
        db = OPTICS(eps=eps, min_samples=min_samples, max_eps=eps).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        dic = {
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters_,
            'n_noise': n_noise_
        }
        if n_clusters_ == 0:
            continue

        dic['homogeneity'] = metrics.homogeneity_score(labels_true, labels)
        dic['completeness'] = metrics.completeness_score(labels_true, labels)
        dic['v_measure'] = metrics.v_measure_score(labels_true, labels)
        dic['ARI'] = metrics.adjusted_rand_score(labels_true, labels)
        dic['AMI'] = metrics.adjusted_mutual_info_score(labels_true, labels)
        dic['Silhouette'] = metrics.silhouette_score(X, labels)
        ldbs.append(dic)

        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )

        plt.title("OPTICS, eps : %.3f, min_samples: %d \nEstimated number of clusters: %d" % (
        eps, min_samples, n_clusters_))
        plt.show()

pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 15)
df = pd.DataFrame(ldbs)
print('\nList of eps values: ', leps)
print('List of min_samples: ', lsamples)
print(df)


# #############################################################################
# Compute DBSCAN
eps = 0.2
min_samples = 73
max_eps = 0.5

db = DBSCAN(eps=0.2, min_samples=10).fit(X)
db = OPTICS(eps=eps, min_samples=min_samples, max_eps=max_eps).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print(
   "Adjusted Mutual Information: %0.3f"
   % metrics.adjusted_mutual_info_score(labels_true, labels)
)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

#Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title("OPTICS, eps : %.3f, min_samples: %d, max_eps: %.3f\nEstimated number of clusters: %d" % (eps, min_samples, max_eps, n_clusters_))
plt.show()