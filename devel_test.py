# from scipy.spatial.distance import squareform
# from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Sample data
# X = np.array([[1, 2], [1, 4], [1, 0],
#               [4, 2], [4, 4], [4, 0]])
#
# # Compute the linkage matrix
# Z = linkage(X, method='ward')

# # Plot dendrogram (for visualization)
# plt.figure(figsize=(10, 7))
# dendrogram(Z)
# plt.title("Dendrogram")
# plt.xlabel('Sample Index')
# plt.ylabel('Distance')
# plt.show()

# # Retrieve cluster labels for a specified number of clusters
# # For example, cutting the dendrogram to have 2 clusters
# max_d = 7  # You can adjust this distance to get the desired number of clusters
# clusters = fcluster(Z, t=max_d, criterion='distance')
#
# # Or directly specifying the number of clusters (e.g., 2 clusters)
# k = 2
# clusters_k = fcluster(Z, t=k, criterion='maxclust')
#
# # `clusters` and `clusters_k` contain the labels of clusters for each point
#
# # Assuming distance_matrix is your precomputed square form distance matrix
# # For example purposes, let's create a mock square distance matrix
# distance_matrix = np.array([
#     [0, 2, 3, 4],
#     [2, 0, 1, 3],
#     [3, 1, 0, 5],
#     [4, 3, 5, 0]
# ])
#
# # Convert the square distance matrix to condensed form
# condensed_distance_matrix = squareform(distance_matrix)
#
# # Use the condensed distance matrix with linkage
# Z = linkage(condensed_distance_matrix, 'single')
#
# # Generate the dendrogram
# plt.figure(figsize=(10, 7))
# dendrogram(Z)
# plt.title("Dendrogram")
# plt.xlabel('Sample Index or (Cluster Size)')
# plt.ylabel('Distance')
# plt.show()

import numpy as np

# Sample data matrix X of shape (n_samples, n_features)
X = np.array([[1, 2], [2, 3], [3, 4]])

# Step 1: Expand X's dimensions to prepare for broadcasting
X_expanded = X[:, np.newaxis, :]

# Step 2: Compute pairwise vector sums using broadcasting
pairwise_sums = X_expanded + X_expanded.transpose(1, 0, 2)

# Step 3: Calculate the norm of each sum
pairwise_distances = np.linalg.norm(pairwise_sums, axis=2)

# The above gives a full square matrix of distances. For linkage, we need the condensed form.
# Condensing the full distance matrix (we only need the upper triangle, excluding the diagonal)
i_upper = np.triu_indices(n=pairwise_distances.shape[0], k=1)
condensed_distance_vector = pairwise_distances[i_upper]

# Now, condensed_distance_vector is ready for use with linkage.


