# Ethomap

Lightweight library with basic modules for generating behavioral embeddings from dynamic time warping of postural time
series. Also includes modules to decompose behavioral sequences into transition modes.

Based on [Mearns et al. (2020)](https://www.sciencedirect.com/science/article/pii/S0960982219314617).

# Installation

```commandline
pip install ethomap
```

# Examples

## Dynamic time warping

```python
from ethomap import DynamicTimeWarping  # from dtw module
import numpy as np

# Create some 3D time series
x = np.random.rand(200, 3)
y = np.random.rand(200, 3)

# Align time series y to x
dtw = DynamicTimeWarping(x)
dist = dtw.align(y)
print("DTW distance between x and y:", dist)
```

## Behavioral embedding

```python
from ethomap import pdist_dtw, affinity_propagation, IsomapPrecomputed
import numpy as np
import time
from scipy.spatial.distance import squareform

# Create some 3D time series to align pairwise
xs = np.random.rand(1000, 50, 3)

# Compute pairwise distances between time series (pdist_dtw from ethomap.distance)
t0 = time.time()
distances = pdist_dtw(xs, parallel_processing=True, n_processors=4, bw=0.1, fs=100)
distance_matrix = squareform(distances)  # make square
t1 = time.time()
print("Time taken:", t1 - t0, "seconds")  # ~30 seconds for data of this size
print(distance_matrix.shape)  # (1000, 1000)

# Create behavioral space from all data (IsomapPrecomputed from ethomap.embed)
isomap = IsomapPrecomputed(n_neighbors=5, n_components=2)
embedding = isomap.fit_transform(distance_matrix)
print(embedding.shape)  # (1000, 2)

# Find exemplars by clustering distance matrix (affinity_propagation from ethomap.cluster)
ap = affinity_propagation(distance_matrix, preference="median")
cluster_labels = ap.labels_
exemplar_indices = ap.cluster_centers_indices_

# Select exemplars from original data
exemplars = xs[exemplar_indices]
exemplar_distances = distance_matrix[exemplar_indices, :][:, exemplar_indices]
print(exemplars.shape)  # (n_exemplars, 50, 3)
print(exemplar_distances.shape)  # (n_exemplars, n_exemplars)

# Create behavioral space from exemplars only
embedding_exemplars = isomap.fit_transform(exemplar_distances)
print(embedding_exemplars.shape)  # (n_exemplars, 2)
```
