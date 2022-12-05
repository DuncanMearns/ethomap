import numpy as np
from numba import njit
from sklearn.metrics import pairwise_distances


@njit
def fast_dtw(cost, bw):
    n, m = cost.shape  # get shape
    # Create DTW matrix and fill first row with initial cost
    dtw = np.zeros_like(cost)
    dtw.fill(np.inf)
    dtw[0, :bw] = cost[0, 0:bw]
    # Main loop of dtw algorithm
    for i in range(1, n):
        for j in range(max(0, i - bw + 1), min(m, i + bw)):
            dtw[i, j] = cost[i, j] + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return dtw


@njit
def fast_dtw_1d(s, t, bw):
    """Calculate the DTW distance matrix for two equal length 1-dimensional time series."""
    # Initialise distance matrix
    n = len(s)
    m = len(t)
    dtw = np.empty((n, m))
    dtw.fill(np.inf)
    # Fill the first row without a cost allowing optimal path to be found starting anywhere within the bandwidth
    dtw[0, :bw] = np.array([np.abs(s[0] - t[j]) for j in range(0, bw)])
    # Main loop of dtw algorithm
    for i in range(1, n):
        for j in range(max(0, i - bw + 1), min(m, i + bw)):
            dtw[i, j] = np.abs(s[i] - t[j]) + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return dtw


class DynamicTimeWarping:
    """Dynamic time warping class.

    Parameters
    ----------
    s0 : np.ndarray
        1D (n_samples,) or 2D (n_samples, n_features) template time series.
    bw : float
        Bandwidth of the Sakoe-Chiba band (in seconds).
    fs : float
        Sampling frequency of the time series (samples per second).
    metric : str
        Metric for calculating initial cost matrix. Default = "euclidean".

    Attributes
    ----------
    s1 : np.ndarray
        1D (n_samples,) or 2D (n_samples, n_features) time series to align.
    t0 : np.ndarray
        Zeros-padded copy of s0.
    t1 : np.ndarray
        Zero-padded copy of s1
    DTW : np.ndarray
        Dynamic time warping distance matrix.
    """

    def __init__(self, s0: np.ndarray = None, bw: float = 0.01, fs: float = 500., metric="euclidean"):
        if s0 is not None:
            self.s0 = np.array(s0)
        else:
            self.s0 = None
        self.bw = bw
        self.fs = fs
        self.metric = metric

    @property
    def ndims(self):
        """Number of dimensions of time series."""
        if self.s0.ndim == 2:
            return self.s0.shape[1]
        else:
            return 1

    @property
    def n(self):
        """Number of frames in time series."""
        return max([len(self.s0), len(self.s1)])

    def align(self, s1, s0=None):
        """Align a time series (s1) to the template series (s0).

        Parameters
        ----------
        s1 : np.ndarray
            1D (n_samples,) or 2D (n_samples, n_features) time series to align. Must be same dimensionality as s0.

        Returns
        -------
        distance : float
            The minimum total cost to align s1 to s0, corresponding to lower right element of the DTW matrix.
        """
        # Update template
        if s0 is not None:
            self.s0 = s0
        assert (self.s0 is not None), 'Must provide a template series, template.'
        # Check number of dimensions
        s1 = np.array(s1)
        assert self.s0.ndim == s1.ndim, 'template and s1 must have same number of dimensions.'
        self.s1 = s1
        # Calculate bandwidth in frames
        bw = int(self.bw * self.fs)
        if self.ndims == 1:
            # Create zero-padded arrays, s and t, to align
            self.s0 = self.s0.squeeze()
            self.s1 = self.s1.squeeze()
            self.t0, self.t1 = np.zeros(self.n), np.zeros(self.n)
            self.t0[:len(self.s0)] = self.s0
            self.t1[:len(self.s1)] = self.s1
            # Calculate distance matrix
            self.DTW = self.dtw_1d(self.t0, self.t1, bw)
        else:
            # Create zero-padded arrays, s and t, to align
            self.t0, self.t1 = np.zeros((self.n, self.ndims)), np.zeros((self.n, self.ndims))
            self.t0[:len(self.s0)] = self.s0
            self.t1[:len(self.s1)] = self.s1
            # Calculate distance matrix
            self.DTW = self.dtw(self.t0, self.t1, bw, metric=self.metric)
        # Get the alignment distance
        distance = self.DTW[-1, -1]
        return distance

    def path(self):
        """Compute the path through the distance matrix that produces the optimal alignment of the two time series.

        Returns
        -------
        i, x : np.ndarray, np.ndarray
            Indices and values of s1 that align to the template s0.
        """
        path = [np.array((self.n - 1, self.n - 1))]
        while ~np.all(path[-1] == (0, 0)):
            steps = np.array([(-1, 0), (-1, -1), (0, -1)]) + path[-1]
            if np.any(steps < 0):
                idxs = np.ones((3,), dtype='bool')
                idxs[np.where(steps < 0)[0]] = 0
                steps = steps[idxs]
            path.append(steps[np.argmin(self.DTW[steps[:, 0], steps[:, 1]])])
        path = np.array(path)[::-1]
        return path[:, 0], self.t1[path[:, 1]]

    def map_to_template(self, *series):
        """Map multiple time series to the template.

        Returns
        -------
        np.ndarray
            Array of distances between series and template.
        """
        return np.array([self.align(s) for s in series])

    @staticmethod
    def dtw(s, t, bw, metric="euclidean"):
        """Calculate the DTW distance matrix for two equal length n-dimensional time series."""
        cost = pairwise_distances(s, t, metric=metric)
        return fast_dtw(cost, bw)

    @staticmethod
    def dtw_1d(s, t, bw):
        """Calculate the DTW distance matrix for two equal length 1-dimensional time series."""
        return fast_dtw_1d(s, t, bw)


if __name__ == "__main__":

    t = np.linspace(0, 2 * np.pi, 500)
    a = np.linspace(-3, 3, 500) ** 2
    s0 = np.array([a * np.sin(5 * t), a * np.cos(5 * t)]).T
    s1 = np.array([1.5 * a[:400] * np.sin(5 * t[:400]), a[:400] * np.cos(5 * (t[:400] + (np.pi / 3)))]).T

    DTW = DynamicTimeWarping(s0, bw=0.05, fs=500.)
    d = DTW.align(s1)
    i, x = DTW.path()

    # from matplotlib import pyplot as plt
    # plt.plot(*s2.T)
    # plt.show()

    # fig, axes = plt.subplots(2, 1)
    # for i in range(2):
    #     axes[i].plot(template[:, i])
    #     axes[i].plot(s1[:, i])
    #     axes[i].plot(s, t[:, i])
    # plt.show()
