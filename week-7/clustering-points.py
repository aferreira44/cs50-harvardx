import matplotlib.pyplot as plt     # for generating plots
import numpy as np                  # for scientific computing

# import the k-means algorithm to cluster the points

from sklearn.cluster import KMeans

%matplotlib notebook

# create data points (unlabeled) to be clustered

X = np.array([[1,1], [2,2.5], [3,1.2], [5.5,6.3], [6,9], [7,6], [8,8]])

# plot them

plt.figure()
plt.scatter(X[:,0], X[:,1], s = 170, color = 'black')
plt.show()

# Run k-means with k = 2

k = 2
kmeans = KMeans(n_clusters = k)
kmeans.fit(X);
centroids = kmeans.cluster_centers_    # Get centroid's coordinates
labels = kmeans.labels_                # Get label assignment

# plot the points with label assignments as given by kmeans

colors= ['r.', 'g.']    # Define two colors for the plot below
plt.figure()
for i in range(len(X)):
    plt.plot(X[i,0], X[i,1], colors[labels[i]], markersize = 30)
plt.scatter(centroids[:,0], centroids[:,1], marker = "x", s = 300, linewidths=5)

# Run k-means with k = 3

k = 3
kmeans = KMeans(n_clusters = k)
kmeans.fit(X);
centroids = kmeans.cluster_centers_    # Get centroid's coordinates
labels = kmeans.labels_                # Get label assignment

# plot the points with label assignments as given by kmeans

colors= ['r.', 'g.', 'y.']    # Define two colors for the plot below
plt.figure()
for i in range(len(X)):
    plt.plot(X[i,0], X[i,1], colors[labels[i]], markersize = 30)
plt.scatter(centroids[:,0], centroids[:,1], marker = "x", s = 300, linewidths=5)

# Run k-means with k = 4

k = 4
kmeans = KMeans(n_clusters = k)
kmeans.fit(X);
centroids = kmeans.cluster_centers_    # Get centroid's coordinates
labels = kmeans.labels_                # Get label assignment

# plot the points with label assignments as given by kmeans

colors= ['r.', 'g.', 'y.', 'c.']    # Define two colors for the plot below
plt.figure()
for i in range(len(X)):
    plt.plot(X[i,0], X[i,1], colors[labels[i]], markersize = 30)
plt.scatter(centroids[:,0], centroids[:,1], marker = "x", s = 300, linewidths=5)

# Run k-means with k = 5

k = 5
kmeans = KMeans(n_clusters = k)
kmeans.fit(X);
centroids = kmeans.cluster_centers_    # Get centroid's coordinates
labels = kmeans.labels_                # Get label assignment

# plot the points with label assignments as given by kmeans

colors= ['r.', 'g.', 'y.', 'c.', 'b.']    # Define two colors for the plot below
plt.figure()
for i in range(len(X)):
    plt.plot(X[i,0], X[i,1], colors[labels[i]], markersize = 30)
plt.scatter(centroids[:,0], centroids[:,1], marker = "x", s = 300, linewidths=5)

# Run k-means with k = 6

k = 6
kmeans = KMeans(n_clusters = k)
kmeans.fit(X);
centroids = kmeans.cluster_centers_    # Get centroid's coordinates
labels = kmeans.labels_                # Get label assignment

# plot the points with label assignments as given by kmeans

colors= ['r.', 'g.', 'y.', 'c.', 'b.', 'k.']    # Define two colors for the plot below
plt.figure()
for i in range(len(X)):
    plt.plot(X[i,0], X[i,1], colors[labels[i]], markersize = 30)
plt.scatter(centroids[:,0], centroids[:,1], marker = "x", s = 300, linewidths=5)

# Run k-means with k = 7

k = 7
kmeans = KMeans(n_clusters = k)
kmeans.fit(X);
centroids = kmeans.cluster_centers_    # Get centroid's coordinates
labels = kmeans.labels_                # Get label assignment

# plot the points with label assignments as given by kmeans

colors= ['r.', 'g.', 'y.', 'c.', 'b.', 'k.', 'm.']    # Define two colors for the plot below
plt.figure()
for i in range(len(X)):
    plt.plot(X[i,0], X[i,1], colors[labels[i]], markersize = 30)
plt.scatter(centroids[:,0], centroids[:,1], marker = "x", s = 300, linewidths=5)