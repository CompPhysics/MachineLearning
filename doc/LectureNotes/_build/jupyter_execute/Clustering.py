<!-- HTML file automatically generated from DocOnce source (https://github.com/doconce/doconce/)
doconce format html Clustering.do.txt  -->

# Clustering Analysis
In this chapter we will concern ourselves with the study of **cluster analysis**.
In general terms cluster analysis, or clustering, is the task of grouping a
data-set into different distinct categories based on some measure of equality of
the data. This measure is often referred to as a **metric** or **similarity
measure** in the literature (note: sometimes we deal with a **dissimilarity
measure** instead). Usually, these metrics are formulated as some kind of
distance function between points in a high-dimensional space.

There exists a lot of such distance measures. The simplest, and also the most
common is the **Euclidean distance** (i.e. Pythagoras). A good source for those of
you wanting a thorough overview is the article (DOI:10.5120/ijca2016907841
Irani, Pise, Phatak). A few other metrics mentioned there are: *cosine
similarity*, *Manhattan distance*, *Chebychev distance* and the *Minkowski
distance*. The Minkowski distance is a general formulation which encapsulates a
range of metrics. All of these, and many more, can be used in clustering. There
exists different categories of clustering algorithms. A few of the most
common are: *centroid-*, *distribution-*, *density-* and *hierarchical-
clustering*. We will concern ourselves primarily with the first one.

## Basic Idea of the K-means Clustering Algorithm
The simplest of all clustering algorithms is the aptly named **k-means algorithm**
, sometimes also referred to as *Lloyds algorithm*. It is the simplest and also
the most common. From its simplicity it obtains both strengths and weaknesses.
These will be discussed in more detail later. The k-means algorithm is a
**centroid based** clustering algorithm.

Assume, we are given $n$ data points and we wish to split the data into $K < n$
different categories, or clusters. We label each cluster by an integer $k\in\{
1, \cdots, K \}$. In the basic k-means algorithm each point is assigned to only
one cluster $k$, and these assignments are *non-injective* i.e. many-to-one. We
can think of these mappings as an encoder $k = C(i)$, which assigns the $i$-th
data-point $\bf x_i$ to the $k$-th cluster. Before we jump into the mathematics
let us describe the k-means algorithm in words:
1. We start with guesses / random initializations of our $k$ cluster centers / centroids

2. For each centroid the points that are most similar are identified

3. Then we move / replace each centroid with a coordinate average of all the points that were assigned to that centroid.

4. Iterate this points 2, 3) until the centroids no longer move (to some tolerance)

Now we consider the method formally. Again, we assume we have $n$ data-points
(vectors)

<!-- Equation labels as ordinary links -->
<div id="eq:kmeanspoints"></div>

$$
\begin{equation}\label{eq:kmeanspoints} \tag{1}
  \boldsymbol{x_i}  = \{x_{i, 1}, \cdots, x_{i, p}\}\in\mathbb{R}^p.
\end{equation}
$$

which we wish to group into $K < n$ clusters. For our dissimilarity measure we
will use the *squared Euclidean distance*

<!-- Equation labels as ordinary links -->
<div id="eq:squaredeuclidean"></div>

$$
\begin{equation}\label{eq:squaredeuclidean} \tag{2}
  d(\boldsymbol{x_i}, \boldsymbol{x_i'}) = \sum_{j=1}^p(x_{ij} - x_{i'j})^2
                         = ||\boldsymbol{x_i} - \boldsymbol{x_{i'}}||^2
\end{equation}
$$

Next we define the so called *within-cluster point scatter* which gives us a
measure of how close each data point assigned to the same cluster tends to be to
the all the others.

<!-- Equation labels as ordinary links -->
<div id="eq:withincluster"></div>

$$
\begin{equation}\label{eq:withincluster} \tag{3}
  W(C) = \frac{1}{2}\sum_{k=1}^K\sum_{C(i)=k}
          \sum_{C(i')=k}d(\boldsymbol{x_i}, \boldsymbol{x_{i'}}) =
          \sum_{k=1}^KN_k\sum_{C(i)=k}||\boldsymbol{x_i} - \boldsymbol{\overline{x_k}}||^2
\end{equation}
$$

where $\boldsymbol{\overline{x_k}}$ is the mean vector associated with the $k$-th
cluster, and $N_k = \sum_{i=1}^nI(C(i) = k)$, where the $I()$ notation is
similar to the Kronecker delta (*Commonly used in statistics, it just means that
when $i = k$ we have the encoder $C(i)$*). In other words,  the within-cluster
scatter measures the compactness of each cluster with respect to the data points
assigned to each cluster. This is the quantity that the $k$-means algorithm aims
to minimize. We refer to this quantity $W(C)$ as the within cluster scatter
because of its relation to the *total scatter*.

<!-- Equation labels as ordinary links -->
<div id="eq:totalscatter"></div>

$$
\begin{equation}\label{eq:totalscatter} \tag{4}
  T = W(C) + B(C) = \frac{1}{2}\sum_{i=1}^n
                    \sum_{i'=1}^nd(\boldsymbol{x_i}, \boldsymbol{x_{i'}})
                  = \frac{1}{2}\sum_{k=1}^K\sum_{C(i)=k}
                    \Big(\sum_{C(i') = k}d(\boldsymbol{x_i}, \boldsymbol{x_{i'}})
                  + \sum_{C(i')\neq k}d(\boldsymbol{x_i}, \boldsymbol{x_{i'}})\Big)
\end{equation}
$$

Which is a quantity that is conserved throughout the $k$-means algorithm. It can
be thought of as the total amount of information in the data, and it is composed
of the aforementioned within-cluster scatter and the *between-cluster scatter*
$B(C)$. In methods such as principle component analysis the total scatter is not
conserved.

Given a cluster mean $\boldsymbol{m_k}$ we define the **total cluster variance**

<!-- Equation labels as ordinary links -->
<div id="eq:totalclustervariance"></div>

$$
\begin{equation}\label{eq:totalclustervariance} \tag{5}
  \min_{C, \{\boldsymbol{m_k}\}_1^K}\sum_{k=1}^KN_k\sum||\boldsymbol{x_i} - \boldsymbol{m_k}||^2
\end{equation}
$$

Now we have all the pieces necessary to formally revisit the k-means algorithm.
If you at this point feel like some of the above definitions came a bit out of
no-where, don't fret, the method does get a whole lot simpler once we start
programming.

## The K-means Clustering Algorithm
The k-means clustering algorithm goes as follows (note in my opinion this
description is a bit complicated and is lifted directly out of ESL HASTIE for
deeper understanding purposes)

1. For a given cluster assignment $C$, and $k$ cluster means $\{m_1, \cdots, m_k\}$. We minimize the total cluster variance with respect to the cluster means $\{m_k\}$ yielding the means of the currently assigned clusters.

2. Given a current set of $k$ means $\{m_k\}$ the total cluster variance is minimized by assigning each observation to the closest (current) cluster mean. That is $$C(i) = \underset{1\leq k\leq K}{\mathrm{argmin}} ||\boldsymbol{x_i} - \boldsymbol{m_k}||^2$$

3. Steps 1 and 2 are repeated until the assignments do not change.

As previously stated the above formulation can be a bit difficult to understand,
*at least the first time*, due to the dense notation used. But all in all the
concept is fairly simple when explained in words. The math needs to be
understood but to help you along the way we summarize the algorithm as follows
(try to look at the terms above to match with the summary).

1. Before we start we specify a number $k$ which is the number of clusters we want to try to separate our data into.

2. We initially choose $k$ random data points in our data as our initial centroids, *or means* (this is where the name comes from).

3. Assign each data point to their closest centroid, based on the squared Euclidean distance.

4. For each of the $k$ cluster we update the centroid by calculating new mean values for all the data points in the cluster.

5. Iteratively minimize the within cluster scatter by performing steps (3, 4) until the new assignments stop changing (can be to some tolerance) or until a maximum number of iterations have passed.

That's it, nothing magical happening.

## Writing Our Own Code
In the following section we will work to develop a deeper understanding of the
previously discussed mathematics through developing codes to do k-means cluster
analysis.

### Basic Python

Let us now program the most basic version of the algorithm using nothing but
Python with numpy arrays. This code is kept intentionally simple to gradually
progress our understanding. There is no vectorization of any kind, and even most
helper functions are not utilized. Throughout our implementation process it will
be helpful to keep in mind both the mathematical description of the algorithm
*and* our summary from above. In addition, try to think of ways to optimize this
while reading the next section. We will get to it, take it as a challenge to see
if your optimizations are better.

First of all we need a dataset to do our cluster analysis on, for clarity (and
lack of googling beforehand) we generate it ourselves using Gaussians. First we
import

%matplotlib inline

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import display

np.random.seed(2021)

Next we define functions, for ease of use later, to generate Gaussians and to
set up our toy data set.

def gaussian_points(dim=2, n_points=1000, mean_vector=np.array([0, 0]),
                    sample_variance=1):
    """
    Very simple custom function to generate gaussian distributed point clusters
    with variable dimension, number of points, means in each direction
    (must match dim) and sample variance.

    Inputs:
        dim (int)
        n_points (int)
        mean_vector (np.array) (where index 0 is x, index 1 is y etc.)
        sample_variance (float)

    Returns:
        data (np.array): with dimensions (dim x n_points)
    """

    mean_matrix = np.zeros(dim) + mean_vector
    covariance_matrix = np.eye(dim) * sample_variance
    data = np.random.multivariate_normal(mean_matrix, covariance_matrix,
                                    n_points)
    return data



def generate_simple_clustering_dataset(dim=2, n_points=1000, plotting=True,
                                    return_data=True):
    """
    Toy model to illustrate k-means clustering
    """

    data1 = gaussian_points(mean_vector=np.array([5, 5]))
    data2 = gaussian_points()
    data3 = gaussian_points(mean_vector=np.array([1, 4.5]))
    data4 = gaussian_points(mean_vector=np.array([5, 1]))
    data = np.concatenate((data1, data2, data3, data4), axis=0)

    if plotting:
        fig, ax = plt.subplots()
        ax.scatter(data[:, 0], data[:, 1], alpha=0.2)
        ax.set_title('Toy Model Dataset')
        plt.show()


    if return_data:
        return data


data = generate_simple_clustering_dataset()

Now that we are our, albeit very simple, dataset we are ready to start
implementing the k-means algorithm.

n_samples, dimensions = data.shape
n_clusters = 4

# we randomly initialize our centroids
np.random.seed(2021)
centroids = data[np.random.choice(n_samples, n_clusters, replace=False), :]
distances = np.zeros((n_samples, n_clusters))

# first we need to calculate the distance to each centroid from our data
for k in range(n_clusters):
    for n in range(n_samples):
        dist = 0
        for d in range(dimensions):
            dist += np.abs(data[n, d] - centroids[k, d])**2
            distances[n, k] = dist

# we initialize an array to keep track of to which cluster each point belongs
# the way we set it up here the index tracks which point and the value which
# cluster the point belongs to
cluster_labels = np.zeros(n_samples, dtype='int')

# next we loop through our samples and for every point assign it to the cluster
# to which it has the smallest distance to
for n in range(n_samples):
    # tracking variables (all of this is basically just an argmin)
    smallest = 1e10
    smallest_row_index = 1e10
    for k in range(n_clusters):
        if distances[n, k] < smallest:
            smallest = distances[n, k]
            smallest_row_index = k

    cluster_labels[n] = smallest_row_index

Let's plot and see

fig = plt.figure()
ax = fig.add_subplot()
unique_cluster_labels = np.unique(cluster_labels)
for i in unique_cluster_labels:
    ax.scatter(data[cluster_labels == i, 0],
               data[cluster_labels == i, 1],
               label = i,
               alpha = 0.2)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='black')

ax.set_title("First Grouping of Points to Centroids")

plt.show()

So what do we have so far? We have 'picked' $k$ centroids at random from our
data points. There are other ways of more intelligently choosing their
initializations, however for our purposes randomly is fine. Then we have
initialized an array 'distances' which holds the information of the distance,
*or dissimilarity*, of every point to of our centroids. Finally, we have
initialized an array 'cluster_labels' which according to our distances array
holds the information of to which centroid every point is assigned. This was the
first pass of our algorithm. Essentially, all we need to do now is repeat the
distance and assignment steps above until we have reached a desired convergence
or a maximum amount of iterations.


max_iterations = 100
tolerance = 1e-8
start_time = time.time()

for iteration in range(max_iterations):
    prev_centroids = centroids.copy()
    for k in range(n_clusters):
        # this array will be used to update our centroid positions
        vector_mean = np.zeros(dimensions)
        mean_divisor = 0
        for n in range(n_samples):
            if cluster_labels[n] == k:
                vector_mean += data[n, :]
                mean_divisor += 1

        # update according to the k means
        centroids[k, :] = vector_mean / mean_divisor

    # we find the dissimilarity
    for k in range(n_clusters):
        for n in range(n_samples):
            dist = 0
            for d in range(dimensions):
                dist += np.abs(data[n, d] - centroids[k, d])**2
                distances[n, k] = dist

    # assign each point
    for n in range(n_samples):
        smallest = 1e10
        smallest_row_index = 1e10
        for k in range(n_clusters):
            if distances[n, k] < smallest:
                smallest = distances[n, k]
                smallest_row_index = k

        cluster_labels[n] = smallest_row_index

    # convergence criteria
    centroid_difference = np.sum(np.abs(centroids - prev_centroids))
    if centroid_difference < tolerance:
        print(f'Converged at iteration {iteration}')
        print(f'Runtime: {time.time() - start_time} seconds')
        break

    elif iteration == max_iterations:
        print(f'Did not converge in {max_iterations} iterations')
        print(f'Runtime: {time.time() - start_time} seconds')

And thats it! We now have an extremely barebones, un-optimized k-means
clustering implementation. Lets plot the final result

fig = plt.figure()
ax = fig.add_subplot()
unique_cluster_labels = np.unique(cluster_labels)
for i in unique_cluster_labels:
    ax.scatter(data[cluster_labels == i, 0],
               data[cluster_labels == i, 1],
               label = i,
               alpha = 0.2)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='black')

ax.set_title("Final Result of K-means Clustering")

plt.show()

Now there are a few glaring improvements to be done here. First of all is
organizing things into functions for better readability. Second is getting rid
of the small inefficiencies like manually calculating distances and argmin. And
finally, we need to optimize for better run-time. It's like we always say: the
best way of looping in Python is to not loop in Python. Let us tackle the first
two improvements.

## Towards a More Numpythonic Code


def get_distances_to_clusters(data, centroids):
    """
    Function that for each cluster finds the squared Euclidean distance
    from every data point to the cluster centroid and returns a numpy array
    containing the distances such that distance[i, j] means the distance between
    the i-th point and the j-th centroid.
    Inputs:
        data (np.array): with dimensions (n_samples x dim)
        centroids (np.array): with dimensions (n_clusters x dim)

    Returns:
        distances (np.array): with dimensions (n_samples x n_clusters)
    """

    n_samples, dimensions = data.shape
    n_clusters = centroids.shape[0]
    distances = np.zeros((n_samples, n_clusters))
    for k in range(n_clusters):
        for i in range(n_samples):
            dist = 0
            for j in range(dimensions):
                dist += np.abs(data[i, j] - centroids[k, j])**2
                distances[i, k] = dist

    return distances



def assign_points_to_clusters(distances):
    """
    Function to assign each data point to the cluster to which it is the closest
    based on the squared Euclidean distance from the get_distances_to_clusters
    method.
    Inputs:
        distances (np.array): with dimensions (n_samples x n_clusters)

    Returns:
        cluster_labels (np.array): with dimensions (n_samples)
    """
    cluster_labels = np.argmin(distances, axis=1)

    return cluster_labels



def k_means(data, n_clusters=4, max_iterations=100, tolerance=1e-8):
    """
    Naive implementation of the k-means clustering algorithm. A short summary of
    the algorithm is as follows: we randomly initialize k centroids / means.
    Then we assign, using the squared Euclidean distance, every data-point to a
    cluster. We then update the position of the k centroids / means, and repeat
    until convergence or we reach our desired maximum iterations. The method
    returns the cluster assignments of our data-points and a sequence of
    centroids.
    Inputs:
        data (np.array): with dimesions (n_samples x dim)
        n_clusters (int): hyperparameter which depends on dataset
        max_iterations (int): hyperparameter which depends on dataset
        tolerance (float): convergence measure

    Returns:
        cluster_labels (np.array): with dimension (n_samples)
        centroid_list (list): list of centroids (np.array)
                              with dimensions (n_clusters x dim)
    """

    samples, dimensions = data.shape
    np.random.seed(2021)
    centroids = data[np.random.choice(len(data), n_clusters, replace=False), :]
    distances = get_distances_to_clusters(data, centroids)
    cluster_labels = assign_points_to_clusters(distances)

    start_time = time.time()

    for iteration in range(max_iterations):
        prev_centroids = centroids.copy()
        for k in range(n_clusters):
            vector_mean = np.zeros(dimensions)
            mean_divisor = 0
            for n in range(n_samples):
                if cluster_labels[n] == k:
                    vector_mean += data[n, :]
                    mean_divisor += 1
            # And update according to the new means
            centroids[k, :] = vector_mean / mean_divisor

        distances = get_distances_to_clusters(data, centroids)
        cluster_labels = assign_points_to_clusters(distances)

        centroid_difference = np.sum(np.abs(centroids - prev_centroids))
        if centroid_difference < tolerance:
            print(f'Converged at iteration: {iteration}')
            print(f'Runtime: {time.time() - start_time} seconds')

            return cluster_labels, centroids

    print(f'Did not converge in {max_iterations} iterations')
    print(f'Runtime: {time.time() - start_time} seconds')

    return cluster_labels, centroids


# quirk of numpy / Jupyter need to set seed again
cluster_labels, centroids = k_means(data)

**Note**: the start of the timing is after the random initialization, and first
'cycle' of our algorithm. This is technically not the correct way to time it but
due to this being in a Jupyter notebook and the way it is structured this way of
comparing our algorithms will produce a more equal result. When timing code we
should always encapsulate our whole computation block.

So we see an improvement from just switching to numpy's argmin function. There
is a very nice tool (or category of tools) called profilers. These can be
utilized to make clearer which improvements to our code we should care most
about here is an [excellent source](https://ipython-books.github.io/42-profiling-your-code-easily-with-cprofile-and-ipython/)
on the topic. Even before optimizing we can understand which parts of our code
will be taking the most of the run-time. It will be the longest Python loop,
i.e. the loop over all the samples. Nonetheless, let us do some profiling!

test_data = generate_simple_clustering_dataset(n_points=10000, plotting=False)
%prun -l 10 cluster_labels, centroids = k_means(test_data)

Here we can see the reason for profiling. We now know for certain a lot can be
gained just by vectorizing our distance function. Ideally we wish to perform
most of our loops in numpy, i.e. C. To do this we need our array shapes to match
and clever reshaping will let us do so.


def np_get_distances_to_clusters(data, centroids):
    """
    Squared Euclidean distance between all data-points and every centroid. For
    the function to work properly it needs data and centroids to be numpy
    broadcastable. We sum along the dimension axis.
    Inputs:
        data (np.array): with dimensions (samples x 1 x dim)
        centroids (np.array): with dimensions (1 x n_clusters x dim)

    Returns:
        distances (np.array): with dimensions (samples x n_clusters)
    """

    distances = np.sum(np.abs((data - centroids))**2, axis=2)
    return distances

def np_assign_points_to_clusters(distances):
    """
    Assigning each data-point to a cluster given an array distances containing
    the squared Euclidean distance from every point to each centroid. We do
    np.argmin along the cluster axis to find the closest cluster. Returns a
    numpy array with corresponding labels.
    Inputs:
        distances (np.array): with dimensions (samples x n_clusters)

    Returns:
        cluster_labels (np.array): with dimensions (samples x None)
    """
    cluster_labels = np.argmin(distances, axis=1)
    return cluster_labels


def np_k_means(data, n_clusters=4, max_iterations=100, tolerance=1e-8):
    """
    Numpythonic implementation of the k-means clusting algorithm.
    Inputs:
        data (np.array): with dimesions (samples x dim)
        n_clusters (int): hyperparameter which depends on dataset
        max_iterations (int): hyperparameter which depends on dataset
        tolerance (float): convergence measure
        progression_plot (bool): activation flag for plotting
    Returns:
        cluster_labels (np.array): with dimension (samples)
        centroid_list (list): list of centroids (np.array)
                              with dimensions (n_clusters x dim)
    """
    n_samples, dimensions = data.shape
    np.random.seed(2021)
    centroids = data[np.random.choice(len(data), n_clusters, replace=False), :]

    distances = np_get_distances_to_clusters(np.reshape(data,
                                            (n_samples, 1, dimensions)),
                                          np.reshape(centroids,
                                            (1, n_clusters, dimensions)))
    cluster_labels = np_assign_points_to_clusters(distances)

    start_time = time.time()

    for iteration in range(max_iterations):
        prev_centroids = centroids.copy()
        for k in range(n_clusters):
            points_in_cluster = data[cluster_labels == k]
            mean_vector = np.mean(points_in_cluster, axis=0)
            centroids[k] = mean_vector

        distances = np_get_distances_to_clusters(np.reshape(data,
                                                (n_samples, 1, dimensions)),
                                              np.reshape(centroids,
                                                (1, n_clusters, dimensions)))
        cluster_labels = np_assign_points_to_clusters(distances)

        centroid_difference = np.sum(np.abs(centroids - prev_centroids))
        if centroid_difference < tolerance:
            print(f'Converged at iteration: {iteration}')
            print(f'Runtime: {time.time() - start_time} seconds')

            return cluster_labels, centroids

    print(f'Did not converge in {max_iterations} iterations')
    print(f'Runtime: {time.time() - start_time} seconds')

    return cluster_labels, centroids

When working towards becoming a data scientist using Python this last step is
arguably one of the most important. Thinking of ways to avoid explicitly looping
by adding dimensions to our arrays in such a way that they become broadcastable
using numpy (also tensorflow and many others). Let us take a look at our the
fruits of our labor.

cluster_labels, centroids = np_k_means(data)

