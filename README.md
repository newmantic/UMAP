# UMAP

UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique that is particularly useful for visualizing high-dimensional data in low-dimensional space (typically 2D or 3D). It is based on manifold learning, which assumes that the data lies on a low-dimensional manifold within a higher-dimensional space.



Manifold: In the context of UMAP, a manifold is a low-dimensional space that represents the intrinsic structure of the data. The goal of UMAP is to project the high-dimensional data into a lower-dimensional space while preserving the structure of the manifold.

k-Nearest Neighbors (k-NN): UMAP begins by finding the k-nearest neighbors for each data point in the high-dimensional space. This helps in capturing the local structure of the data.

Fuzzy Simplicial Set: UMAP constructs a fuzzy topological representation of the data by converting the k-NN graph into a fuzzy simplicial set. This representation captures both the local and global structure of the data.

Optimization: UMAP then optimizes the fuzzy simplicial set to create a low-dimensional embedding. The optimization process minimizes the difference between the fuzzy set representation in the high-dimensional space and the low-dimensional space.



Input Data: Let X be a dataset consisting of n points in d-dimensional space. Each data point is represented as x_i, where x_i is a vector in R^d.

k-Nearest Neighbors: For each point x_i, UMAP finds the k nearest neighbors in the dataset based on a chosen distance metric (e.g., Euclidean distance). Let NN_k(x_i) represent the set of k nearest neighbors of x_i.

Fuzzy Simplicial Set: UMAP converts the k-NN graph into a weighted graph where the edge weights are determined by a fuzzy set membership function. The weight w_ij between points x_i and x_j is defined as:
w_ij = exp(-(d_ij - rho_i) / sigma_i)
where:
d_ij is the distance between x_i and x_j.
rho_i is the distance to the closest neighbor of x_i, ensuring that very close neighbors are weighted more heavily.
sigma_i is a normalization factor that controls the spread of the neighborhood around x_i.
The fuzzy simplicial set can be thought of as a weighted adjacency matrix W where W[i, j] = w_ij.

Low-Dimensional Embedding: Let Y be the low-dimensional embedding of X, where each point y_i is a vector in R^m and m << d (typically m = 2 or m = 3 for visualization). The goal is to minimize the cross-entropy between the fuzzy set representation in the high-dimensional space and the low-dimensional space.


The optimization objective is:
C(Y) = sum_{i,j} W[i, j] * log((1 + ||y_i - y_j||^2)^(-1)) + (1 - W[i, j]) * log(1 - (1 + ||y_i - y_j||^2)^(-1))
where:
||y_i - y_j||^2 is the squared Euclidean distance between the low-dimensional points y_i and y_j.
The first term ensures that points close in the high-dimensional space remain close in the low-dimensional space. The second term prevents points that are distant in the high-dimensional space from becoming too close in the low-dimensional space.

Stochastic Gradient Descent (SGD): UMAP uses stochastic gradient descent to minimize the objective function C(Y). The algorithm iteratively updates the positions of the points in the low-dimensional space to achieve a good approximation of the manifold.


UMAP can be summarized by the following steps:
Compute k-Nearest Neighbors: Identify the k nearest neighbors for each point in the dataset.
Construct Fuzzy Simplicial Set: Build a weighted graph (fuzzy simplicial set) representing the local relationships in the data.
Optimize Embedding: Use an optimization algorithm (typically SGD) to find a low-dimensional representation that preserves both the local and global structure of the data.
