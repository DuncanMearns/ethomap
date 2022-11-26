import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import expon


def transition_matrix(seq, index=None, n_states=None):
    """Compute a transition matrix from a sequence of states.

    Parameters
    ----------
    seq : np.array
        Sequence of states
    index : np.array
        Sequence of indices for given states
    n_states : int
        Total number of states

    Returns
    -------
    T : np.array, shape: (n_states, n_states)
        Number of transitions between each pair of states. T[i, j] = n_transitions(j -> i)
    """
    if n_states is None:
        n_states = seq.max() + 1
    if index is None:
        index = np.arange(len(seq))
    T = np.zeros((n_states, n_states))
    t0 = np.where(np.diff(index) == 1)[0]
    transitions = np.array([seq[t0], seq[t0 + 1]]).T
    for s0, s1 in transitions:
        T[s1, s0] += 1

    return T


def transition_modes(M):
    """Finds the singular value decomposition of the symmetric and antisymmetric components of matrix M.

    Parameters
    ----------
    M : np.ndarray
        Transition frequency matrix. M[i, j] = n_transitions(j -> i)

    Returns
    -------
    USVs, USVa : np.ndarray
        Symmetric and antisymmetric transitions represented by their SVDs.
    """
    # Compute symmetric and antisymmetric components of matrix, M
    S = 0.5 * (M + M.T)
    A = 0.5 * (M - M.T)
    # Singular value decomposition
    Us, ss, VsT = np.linalg.svd(S)
    Ua, sa, VaT = np.linalg.svd(A)
    # Transpose input vectors
    Vs = VsT.T
    Va = VaT.T
    # Singular values of symmetric component
    Ds = np.zeros(Us.shape)
    Ds[np.diag_indices(len(Ds))] = ss
    # Singular values of antisymmetric component
    Da = np.zeros(Ua.shape)
    Da[np.diag_indices(len(Da))] = sa
    # Combine matrices
    USVs = np.array([Us, Ds, Vs])
    USVa = np.array([Ua, Da, Va])
    return USVs, USVa


def weight_matrix(x: np.ndarray, metric='euclidean', scale=40.):
    """Generate a weight matrix using an inverse exponential kernel.

    Parameters
    ----------
    x : array_like
        Feature vectors (n_samples, n_features) or condensed distance matrix (if metric='precomputed').
    metric : str
        If 'precomputed', x is assumed to be a condensed distance matrix. Otherwise, passed to
        scipy.spatial.distance.pdist.
    scale : float
        Passed to scipy.stats.expon.
    """
    if metric != 'precomputed':
        # Calculate pairwise distances between points
        x = pdist(x, metric=metric)
    # Generates weights
    weight_generator = expon(loc=0, scale=scale)
    weights = weight_generator.pdf(x)
    W = squareform(weights)
    W[np.arange(len(W)), np.arange(len(W))] = weight_generator.pdf(0)
    W = W / W.sum(axis=0)
    return W


def redistribute_transitions(T: np.ndarray, W: np.ndarray):
    """Re-distribute transitions in transition frequency matrix, T, according to some re-weighting matrix, W."""
    if T.ndim == 2:
        return np.dot(np.dot(W, T), W.T)
    else:
        return np.array([np.dot(np.dot(W, t), W.T) for t in T])
