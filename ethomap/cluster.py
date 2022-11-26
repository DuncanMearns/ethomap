import typing
import numpy as np
from sklearn.cluster import AffinityPropagation
from scipy.cluster import hierarchy as sch


_preference_funcs = {
    "min": np.min,
    "minimum": np.min,
    "max": np.max,
    "maximum": np.max,
    "med": np.median,
    "median": np.median
}


def affinity_propagation(D: np.ndarray, preference: typing.Union[str, callable, float] = 'median'):
    """Perform affinity propagation on a precomputed distance matrix.

    Parameters
    ----------
    D : np.ndarray
        Square distance matrix.
    preference : str or callable or float
        Preference for affinity propagation.
        If float, use this value as the preference.
        If str, use look up to find function to compute preference from -D {'min', 'max', 'median'}.
        If callable, compute preference from -D.

    Returns
    -------
    AffinityPropagation
        Instance of sklean.cluster.AffinityPropagation.
    """
    assert D.shape[0] == D.shape[1], 'Matrix is not square!'
    if isinstance(preference, float):
        preference = preference
    else:
        # Get preference function
        try:
            func = _preference_funcs[preference]
        except KeyError:
            func = preference
        # Calculate preference
        preference = func(-D)
    clusterer = AffinityPropagation(affinity='precomputed', preference=preference)
    clusterer.fit_predict(-D)
    return clusterer


def hierarchical_clustering(isomap, n_clusters):
    Z = sch.linkage(isomap, 'ward')
    exemplar_cluster_labels = sch.fcluster(Z, n_clusters, criterion='maxclust')
    exemplar_cluster_labels -= 1
    return exemplar_cluster_labels
