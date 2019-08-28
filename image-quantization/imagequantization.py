#
# Image Quantization
#
# Author: Nasheb Ismaily
#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import mahotas as mh
from sklearn.metrics import silhouette_score,silhouette_samples
import statistics

original_img = np.array(mh.imread('manycolors.jpg'),
                        dtype=np.float64) / 255
width, height, depth = tuple(original_img.shape)
flattened_image = np.reshape(original_img, (width * height, depth))
print(flattened_image)

image__sample = shuffle(flattened_image, random_state=0)[:1000]

Sum_of_squared_distances = []
K = range(1,50)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(image__sample)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

K = range(2,3)

for k in K:
    clusterer = KMeans (n_clusters=k)
    preds = clusterer.fit_predict(image__sample)
    centers = clusterer.cluster_centers_

    score = silhouette_score (image__sample, preds, metric='euclidean')
    print ("For n_clusters = {}, silhouette score is {})".format(k, score))

K = range(2,100)
for k in K:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    km = KMeans(n_clusters=k)
    labels = km.fit_predict(image__sample)
    centroids = km.cluster_centers_

    # Get silhouette samples
    silhouette_vals = silhouette_samples(image__sample, labels)

    # Silhouette plot
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # Get the average silhouette score and plot it
    avg_score = np.mean(silhouette_vals)
    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
    ax1.set_yticks([])
    ax1.set_xlim([-0.1, 1])
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster labels')
    ax1.set_title('Silhouette plot for the various clusters', y=1.02);

    # Scatter plot of data colored with labels
    ax2.scatter(image__sample[:, 0], image__sample[:, 1], c=labels)
    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)
    ax2.set_xlim([-2, 2])
    ax2.set_xlim([-2, 2])
    ax2.set_xlabel('Eruption time in mins')
    ax2.set_ylabel('Waiting time to next eruption')
    ax2.set_title('Visualization of clustered data', y=1.02)
    ax2.set_aspect('equal')
    plt.tight_layout()
    plt.suptitle(f'Silhouette analysis using k = {k}',
                 fontsize=16, fontweight='semibold', y=1.05);


Sum_of_squared_distances = []
K = range(1,50)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(flattened_image)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

sse = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(flattened_image)
    sse.append(km.inertia_)

plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')
plt.show()

color_estimator = KMeans(n_clusters=33, random_state=0)
color_estimator.fit(image__sample)

cluster_assignments = color_estimator.predict(flattened_image)

compressed_palette = color_estimator.cluster_centers_
plt.scatter(flattened_image[:, 0], flattened_image[:, 1], c=cluster_assignments, s=5, cmap='viridis')
plt.scatter(compressed_palette[:, 0], compressed_palette[:, 1], c='black', s=200, alpha=0.5)
plt.show()

compressed_img = np.zeros((width, height, compressed_palette.shape[1]))
label_index = 0
for i in range(width):
    for j in range(height):
        compressed_img[i][j] = compressed_palette[cluster_assignments[label_index]]
        label_index += 1

width, height, depth = tuple(compressed_img.shape)
compressed_image = np.reshape(compressed_img, (width * height, depth))

flat_r = flattened_image[:,0]
flat_g = flattened_image[:,1]
flat_b = flattened_image[:,2]

compress_r = compressed_image[:,0]
compress_g = compressed_image[:,1]
compress_b = compressed_image[:,2]

r_difference = np.subtract(flat_r,compress_r)
g_difference = np.subtract(flat_g,compress_g)
b_difference = np.subtract(flat_b,compress_b)

print("RED")
print('Mean: %s, Median: %s, Min: %s, Max %s' % (statistics.mean(r_difference),statistics.median(r_difference),min(r_difference),max(r_difference)))

print("GREEN")
print('Mean: %s, Median: %s, Min: %s, Max %s' % (statistics.mean(g_difference),statistics.median(g_difference),min(g_difference),max(g_difference)))

print("BLUE")
print('Mean: %s, Median: %s, Min: %s, Max %s' % (statistics.mean(b_difference),statistics.median(b_difference),min(b_difference),max(b_difference)))
