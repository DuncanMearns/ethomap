import numpy as np
from numba import njit
from sklearn.metrics import pairwise_distances


@njit
def fast_dtw(cost, bw):
    """Perform dynamic time warping algorithm with given cost matrix and Sakoe-Chiba band width."""
    n, m = cost.shape  # get shape
    # Create dtw_ matrix and fill first row with initial cost
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
    """Calculate the dtw_ distance matrix for two equal length 1-dimensional time series."""
    assert s.ndim == t.ndim == 1
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
    x, y : np.ndarray
        1D (n_samples,) or 2D (n_samples, n_features) template time series (x) and time series to align (y).
    bw : float
        Bandwidth of the Sakoe-Chiba band (in seconds).
    fs : float
        Sampling frequency of the time series (samples per second).
    metric : str
        Metric for calculating initial cost matrix. Default = "euclidean".
    pad : bool
        If True (default), pads the shorter series with zeros so that x and y are of equal length.

    Attributes
    ----------
    x_ : np.ndarray
        Zero-padded copy of x.
    y_ : np.ndarray
        Zero-padded copy of y
    dtw_ : np.ndarray
        Dynamic time warping distance matrix.
    """

    def __init__(self,
                 x: np.ndarray = None,
                 y: np.ndarray = None,
                 bw: float = 0.01,
                 fs: float = 500.,
                 metric="euclidean",
                 pad=True):
        self.x = np.array(x) if x is not None else x
        self.y = np.array(y) if y is not None else y
        self.bw = bw
        self.fs = fs
        self.metric = metric
        self.pad = pad

    @property
    def ndims(self):
        """Number of dimensions of time series."""
        if self.x.ndim == 2:
            return self.x.shape[1]
        else:
            return 1

    @property
    def max_n(self):
        """Max length of x and y."""
        return max([len(self.x), len(self.y)])

    def align(self, y, x=None):
        """Align a time series (y) to the template series (x).

        Parameters
        ----------
        y : np.ndarray
            1D (n_samples,) or 2D (n_samples, n_features) time series to align. Must be same dimensionality as x.

        Returns
        -------
        distance : float
            The minimum total cost to align y to x, corresponding to lower right element of the dtw_ matrix.
        """
        # Update template
        if x is not None:
            self.x = x
        assert (self.x is not None), 'Must provide a template series, x.'
        # Check number of dimensions
        y = np.array(y)
        assert self.x.ndim == y.ndim, 'x and y must have same number of dimensions.'
        self.y = y
        # Calculate bandwidth in frames
        bw = int(self.bw * self.fs)
        # Create arrays to align
        if self.pad:
            self.x_ = _pad_to_length(self.x, self.max_n)
            self.y_ = _pad_to_length(self.y, self.max_n)
        else:
            self.x_ = self.x
            self.y_ = self.y
        # Calculate distance matrix
        if self.ndims == 1:
            self.dtw_ = self.dtw_1d(self.x_, self.y_, bw)
        else:
            self.dtw_ = self.dtw(self.x_, self.y_, bw, metric=self.metric)
        # Get the alignment distance
        distance = self.dtw_[-1, -1]
        return distance

    def path(self):
        """Compute the path through the distance matrix that produces the optimal alignment of the two time series.

        Returns
        -------
        i, x : np.ndarray, np.ndarray
            Indices and values of y that align to the template x.
        """
        path = [np.array((len(self.x_) - 1, len(self.y_) - 1))]
        while ~np.all(path[-1] == (0, 0)):
            steps = np.array([(-1, 0), (-1, -1), (0, -1)]) + path[-1]
            if np.any(steps < 0):
                idxs = np.ones((3,), dtype='bool')
                idxs[np.where(steps < 0)[0]] = 0
                steps = steps[idxs]
            path.append(steps[np.argmin(self.dtw_[steps[:, 0], steps[:, 1]])])
        path = np.array(path)[::-1]
        return path[:, 0], self.y_[path[:, 1]]

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
        """Calculate the dtw_ distance matrix for two equal length n-dimensional time series."""
        cost = pairwise_distances(s, t, metric=metric)
        return fast_dtw(cost, bw)

    @staticmethod
    def dtw_1d(s, t, bw):
        """Calculate the dtw_ distance matrix for two equal length 1-dimensional time series."""
        return fast_dtw_1d(s, t, bw)


def _pad_to_length(x: np.ndarray, length: int, pad_with: float = 0) -> np.ndarray:
    """Pad end of 1D or 2D array with constant value so that it is of the given length."""
    to_pad = length - len(x)
    if to_pad <= 0:
        return x
    if x.ndim == 1:
        return np.pad(x, (0, to_pad), constant_values=pad_with)
    return np.pad(x, ((0, to_pad), (0, 0)), constant_values=pad_with)


def _test_nd():
    # Create test data
    t = np.linspace(0, 2 * np.pi, 500)
    a = np.linspace(-3, 3, 500) ** 2
    s0 = np.array([a * np.sin(5 * t), a * np.cos(5 * t)]).T
    s1 = np.array([1.5 * a[:400] * np.sin(5 * t[:400]), a[:400] * np.cos(5 * (t[:400] + (np.pi / 3)))]).T
    # Test alignment and path
    DTW = DynamicTimeWarping(s0, bw=0.05, fs=500.)
    d = DTW.align(s1)
    i, x = DTW.path()
    return d, i, x


def _test_1d():
    x = np.random.rand(500)
    y = np.random.rand(300)
    DTW = DynamicTimeWarping(x, bw=50, fs=1)
    d = DTW.align(y)
    i, x = DTW.path()
    return d, i, x
