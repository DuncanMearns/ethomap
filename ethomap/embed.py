from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse.csgraph import shortest_path


class IsomapPrecomputed(Isomap):
    """Hack of sklearn.manifold.Isomap that fixes bug when metric='precomputed'.

    Parameters
    ----------
    weights : np.ndarray, default None
        Square matrix with same shape as x. If provided, distances in the k neighbors graph are scaled by W before
        calculating the geodesic.
    """

    def __init__(self, weights=None, **kwargs):
        super().__init__(metric="precomputed", **kwargs)
        self.weights = None
        if weights:
            self.weights = weights

    def _fit_transform(self, X):
        assert X.shape[0] == X.shape[1]
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.neighbors_algorithm,
                                      metric='precomputed',
                                      n_jobs=self.n_jobs)
        self.nbrs_.fit(X)
        self.training_data_ = self.nbrs_._fit_X
        self.kernel_pca_ = KernelPCA(n_components=self.n_components,
                                     kernel="precomputed",
                                     eigen_solver=self.eigen_solver,
                                     tol=self.tol, max_iter=self.max_iter,
                                     n_jobs=self.n_jobs)

        kng = kneighbors_graph(self.nbrs_, self.n_neighbors, metric='precomputed',
                               mode='distance', n_jobs=self.n_jobs)

        # Hack here (kng needs to be converted to array)
        kng = kng.toarray()
        if self.weights is not None:  # allow rescaling of weights in the kng
            kng = kng * self.weights

        self.dist_matrix_ = shortest_path(kng,
                                          method=self.path_method,
                                          directed=False)
        G = self.dist_matrix_ ** 2
        G *= -0.5

        self.embedding_ = self.kernel_pca_.fit_transform(G)
