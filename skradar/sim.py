import numpy as np
from scipy.constants import speed_of_light as c0


def sim_FMCW_if(r: float, B: float, fc: float, N: float, Ts: float,
                v: float = 0, c: float = c0,
                cplx: bool = False) -> np.ndarray:
    """
    This function simulates the intermediate frequency (IF) signal of a
    single-chirp of an FMCW radar.
    
    Important notes:
        * It is assumed that the chirp duration :math:`T = (N-1)T_s`.
        * Range and velocity have to be entered as round-trip values!

    Parameters
    ----------
    r : float
        Round-trip range in meters (for monostatic radars this is twice the
        target distance).
    B : float
        Bandwidth in Hertz.
    fc : float
        Center frequency in Hertz.
    N : float
        Number of samples.
    Ts : float
        Sample interval (inverse of sampling rate) in seconds.        
    v : float, optional
        Target velocity in meters/second. For monostatic radars this is twice
        the target velocity. The default is 0.
    c : float, optional
        Wave velocity. The default is c0.
    cplx: bool, optional
        Generate a complex-valued signal. The default is False.

    Returns
    -------
    s_if : ndarray
        Samples of the IF signal.

    """
    n = np.arange(N)
    t = n*Ts
    T = t[-1]
    k_ramp = B/T
    tau = (r+v*t)/c  # round-trip time
    if cplx:
        s_if = np.exp(1j*2*np.pi*(fc*tau+k_ramp*tau*t))
    else:
        s_if = np.cos(2*np.pi*(fc*tau+k_ramp*tau*t))
    return s_if
