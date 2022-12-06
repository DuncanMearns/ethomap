import typing
import numpy as np
from joblib import Parallel, delayed

from .dtw import DynamicTimeWarping


def distances_to_template(template: np.ndarray, *series: np.ndarray,
                          bw: float = 0.01, fs: float = 500., include_flip: bool = False, **kwargs) -> np.ndarray:
    """Compute the dtw distances between multiple time series and a template, x.

    Parameters
    ----------
    template : array_like
        1D (n_samples,) or 2D (n_samples, n_features) template time series.
    series : array_like
        Time series to align. Should be same dimensionality as template.
    bw : float
        Bandwidth of the Sakoe-Chiba band (in seconds).
    fs : float
        Sampling frequency of the time series (samples per second).
    include_flip : bool, default False
        If True, takes minimum of [distance(x, y), distance(-x, y)].

    Returns
    -------
    distances : np.ndarray
        Array of dtw distances between series and template (same length as number of series).

    See Also
    --------
    dtw.DynamicTimeWarping
    """
    dtw = DynamicTimeWarping(template, bw, fs)
    distances = dtw.map_to_template(*series)
    if not include_flip:
        return distances
    dtw_flipped = DynamicTimeWarping(-template, bw, fs, **kwargs)
    distances_flipped = dtw_flipped.map_to_template(*series)
    return np.min([distances, distances_flipped], axis=0)


def pdist_dtw(series: typing.Iterable[np.ndarray], parallel_processing: bool = True, n_processors: int = -1, **kwargs):
    """Compute the pairwise dtw distances between series.

    Parameters
    ----------
    series : iterable of array_like
        Time series to align.
    parallel_processing : bool, default True
        If True, compute each row of the distance matrix in parallel.
    n_processors : int, default -1
        Number of processors to use if parallel_processing is True. Default (-1) uses all available processors.
    kwargs
        Passed to multi_dtw.

    Returns
    -------
    D : np.ndarray
        Condensed distance matrix.

    See Also
    --------
    multi_dtw, dtw.DynamicTimeWarping, scipy.spatial.distance.pdist, scipy.spatial.distance.squareform
    """
    series = list(series)
    series_by_row = [series[i:] for i in range(len(series) - 1)]
    if parallel_processing:
        distances = Parallel(n_processors)(delayed(distances_to_template)(*row, **kwargs) for row in series_by_row)
    else:
        distances = [distances_to_template(*row, **kwargs) for row in series_by_row]
    D = np.array([d for row in distances for d in row])
    return D


def cdist_dtw(templates: typing.Iterable[np.ndarray], series: typing.Iterable[np.ndarray],
              parallel_processing: bool = True, n_processors: int = -1, **kwargs):
    """Compute the dtw distances between a set of templates and series.

    Parameters
    ----------
    templates : iterable of array_like
        Template time series to align to.
    series : iterable of array_like
        Time series to be aligned to templates.
    parallel_processing : bool, default True
        If True, compute each row of the distance matrix in parallel.
    n_processors : int, default -1
        Number of processors to use if parallel_processing is True. Default (-1) uses all available processors.
    kwargs
        Passed to multi_dtw.

    Returns
    -------
    D : np.ndarray
        dtw_ distance between each series and template. D[i, j] contains distance between series[i] and template[j].

    See Also
    --------
    multi_dtw, dtw.DynamicTimeWarping, scipy.spatial.distance.cdist
    """
    series = list(series)
    templates = list(templates)
    series_by_row = [[s] + templates for s in series]
    if parallel_processing:
        distances = Parallel(n_processors)(delayed(distances_to_template)(*row, **kwargs) for row in series_by_row)
    else:
        distances = [distances_to_template(*row, **kwargs) for row in series_by_row]
    D = np.array(distances)
    return D
