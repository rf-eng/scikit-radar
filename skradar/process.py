import numpy as np
from scipy.constants import speed_of_light as c0
from skradar import nextpow2
import scipy.spatial


def range_compress_FMCW(s_if: np.ndarray, win_range: np.ndarray, B: float,
                        zp_fact: float, c: float = c0,
                        flatten_phase: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs range-compression on the intermediate frequency (IF) data of an
    FMCW radar and returns the complex-valued range profile together with an
    array containing the round-trip ranges of each entry of the range profile.

    Parameters
    ----------
    s_if : np.ndarray
        The intermediate frequency (IF) data of an FMCW radar. The function 
        sim_FMCW_if can be used to simulate such a signal.
    win_range : np.ndarray, shape(s_if.shape[-1])
        Window function applied along fast-time to control the sidelobes in the
        range profile.
    B : float
        Bandwidth in Hertz.
    zp_fact : float
        Zero-padding factor. The IF signal is zero-padded to
        2**nextpow2(zp_fact*N) with N being the number of samples in s_if.
    c : float, optional
        Wave velocity. The default is c0.
    flatten_phase : bool, optional
        If set to True the linear phase caused by the non-symmetric definition
        of the DFT is removed. The default is True (requires some additional
        calculations).

    Returns
    -------
    range_profile : np.ndarray
        The range profile(s) calculated by applying an fft along the last axis
        of the zero-padded and windowed s_if.
    ranges : np.ndarray
        Round trip ranges for each entry of the range profile(s).

    """
    N = s_if.shape[-1]
    z = 2**nextpow2(zp_fact * N)
    psi = np.fft.fftfreq(z)  # frequency normalized to sampling rate
    if flatten_phase:
        # make (non-zeropadded) time-domain signal symmetric to
        # remove linear phase
        phase_corr = np.exp(1j * 2 * np.pi * psi * (N - 1) / 2)
    else:
        phase_corr = 1
    range_profile = np.fft.fft(s_if * win_range, z) * phase_corr
    ranges = np.linspace(0, 1 - 1 / z, z) * (N - 1) * c / B
    return range_profile, ranges


def backprojection(x_mat, y_mat, z_mat, ant_pos, ant_idcs,
                   px_idcs, rp, rangevec, kw, N_f, posaxis=(0, 1), rangeaxis=-1):

    pixel_coord = np.block(
        [[x_mat.ravel()], [y_mat.ravel()], [z_mat.ravel()]])

    num_pixels = pixel_coord.shape[1]

    tx_pos = ant_pos[0]
    rx_pos = ant_pos[1]
    tx_idcs = ant_idcs[0]
    rx_idcs = ant_idcs[1]

    # calculate distance matrix from each tx to each image pixel:
    pathlens_tx = scipy.spatial.distance_matrix(
        pixel_coord.T, tx_pos.T, threshold=int(1e9))
    # select threshold according to you memory to allow fast computation with large temp arrays
    # could also use sklearn.metrics.pairwise_distances or sklearn.metrics.pairwise.euclidean_distances
    # calculate distance matrix from each rx to each image pixel:
    pathlens_rx = scipy.spatial.distance_matrix(
        pixel_coord.T, rx_pos.T, threshold=int(1e9))
    if np.max(pathlens_tx) + np.max(pathlens_rx) > np.max(rangevec):
        raise ValueError('Image size too large: Range profile does not ' +
                         'cover the largest distance TX->pixel->RX.')

    # calculate round-trip times tx->pixel->rx
    pathlens = pathlens_tx[px_idcs, tx_idcs] + \
        pathlens_rx[px_idcs, rx_idcs]

    # Find indices for range profile (nearest neighbor)
    delta_r = rangevec[1] - rangevec[0]
    pathlen_idcs = np.rint(pathlens / delta_r).astype(int)

    #use tuple for flexible indexing
    idcs_list = [slice(0, None)]*rp.ndim
    if len(posaxis) == 2:  # mimo mode
        idcs_list[posaxis[0]] = tx_idcs
        idcs_list[posaxis[1]] = rx_idcs
        idcs_list[rangeaxis] = pathlen_idcs
    elif len(posaxis) == 1:  # sar mode
        idcs_list[posaxis[0]] = tx_idcs
        idcs_list[rangeaxis] = pathlen_idcs
    elif type(posaxis) is int:  # sar mode
        idcs_list[posaxis] = tx_idcs
        idcs_list[rangeaxis] = pathlen_idcs
    else:
        raise TypeError(
            "posaxis must be either int or a tuple/list/... of length one or two")
    rp_vals = rp[tuple(idcs_list)].astype(np.complex64)
    flat_phase_correction = np.exp(-1j * 2 * np.pi * (pathlens /
                                   rangevec[-1]) * (N_f - 1) / 2).astype(np.complex64)
    phase_corr = np.exp(-1j * kw * pathlens).astype(np.complex64) * \
        flat_phase_correction
    # newaxis to support multiple chirps
    # rp_corr = rp_vals * phase_corr[:, np.newaxis]
    rp_corr = rp_vals.squeeze() * phase_corr
    # unravel
    M_tx = len(np.unique(tx_idcs))
    M_rx = len(np.unique(rx_idcs))
    if len(posaxis) == 2:  # mimo mode
        rp_tmp = rp_corr.reshape(
            (M_tx, M_rx, -1, num_pixels))
        # sum over antennas for each pixel
        image = np.sum(rp_tmp, axis=posaxis)
    elif len(posaxis) == 1:  # sar mode
        rp_tmp = rp_corr.reshape((M_tx, -1))
        # sum over measurement positions for each pixel
        image = np.sum(rp_tmp, axis=posaxis[0])
    elif type(posaxis) is int:  # sar mode
        rp_tmp = rp_corr.reshape((M_tx, -1))
        # sum over measurement positions for each pixel
        image = np.sum(rp_tmp, axis=posaxis)
    return image
