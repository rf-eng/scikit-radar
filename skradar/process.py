import numpy as np
from scipy.constants import speed_of_light as c0
from skradar import nextpow2

def range_compress_FMCW(s_if: np.ndarray, B: float, zp_fact: float,
                        c: float = c0, flatten_phase: bool = True):
    """
    Performs range-compression on the intermediate frequency (IF) data of an
    FMCW radar and returns the complex-valued range profile together with an
    array containing the round-trip ranges of each entry of the range profile.

    Parameters
    ----------
    s_if : np.ndarray
        The intermediate frequency (IF) data of an FMCW radar. The function 
        sim_FMCW_if can be used to simulate such a signal.
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
    z = 2**nextpow2(zp_fact*N)
    psi = np.fft.fftfreq(z)  # frequency normalized to sampling rate
    if flatten_phase:
        # make (non-zeropadded) time-domain signal symmetric to
        # remove linear phase
        phase_corr = np.exp(1j*2*np.pi*psi*(N-1)/2)
    else:
        phase_corr = 1
    range_profile = np.fft.fft(s_if, z)*phase_corr
    ranges = np.linspace(0, 1-1/z, z)*(N-1)*c/B
    return range_profile, ranges
