import numpy as np
from numpy.random import default_rng
from scipy.constants import speed_of_light as c0
from scipy.constants import Boltzmann as kb


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
    t = n * Ts
    T = t[-1]
    k_ramp = B / T
    tau = (r + v * t) / c  # round-trip time
    if cplx:
        s_if = np.exp(1j * 2 * np.pi * (fc * tau + k_ramp * tau * t))
    else:
        s_if = np.cos(2 * np.pi * (fc * tau + k_ramp * tau * t))
    return s_if


def AWGN(N: int, fs: float, T: float = 290, seed: int = None, cplx: bool = False) -> np.ndarray:
    """Generates samples of either real- or complex-valued white Gaussian noise.

    The noise power density is calculated from kb*t using Boltzmann's constant kb. The equivalent
    noise bandwidth is always set to fs/2, also if complex-valued noise is generated.

    Parameters
    ----------
    N : int
        Number of samples.
    fs : float
        Sampling rate in Hz.
    T : float, optional
        Equivalent noise temperature in Kelvin, by default 290 K
    seed : int, optional
        Seed for the random number generator, by default None
    cplx : bool, optional
        Generate complex-valued noise, by default False

    Returns
    -------
    np.ndarray
        Noise samples
    """
    rng = default_rng(seed)
    p_noise = kb * T * fs / 2  # noise power in ADC bandwidth
    if cplx:
        noise = np.sqrt(p_noise) * (rng.standard_normal(N) +
                                    1j * rng.standard_normal(N))
    else:
        noise = np.sqrt(p_noise) * rng.standard_normal(N)
    return noise


def radar_eq(r_tx: float, r_rx: float, rcs: float, lambd: float,
             P_tx: float = 1e-3, G_tx: float = 1, G_rx: float = 1) -> float:
    """Calculates the power at a receiver according to the radar equation.

    Parameters
    ----------
    r_tx : float
        Distance between transmitter and target in meters.
    r_rx : float
        Distance between receiver and target in meters.
    rcs : float
        Radar cross section of the target in m**2.
    lambd : float
        Wavelength in meters.
    P_tx : float, optional
        Transmitter power at the input of the TX antenna in Watts, by default 1e-3 W
    G_tx : float, optional
        Gain of the transmitting antenna (ratio, i.e. not in dBi), by default 1
    G_rx : float, optional
        Gain of the receiving antenna (ratio, i.e. not in dBi), by default 1

    Returns
    -------
    float
        Power at the output of the receiving antenna.
    """
    # power density at target:
    s1 = P_tx * G_tx / (4 * np.pi * r_tx**2)
    # reflected power:
    p_refl = s1 * rcs
    # power density at receiver:
    s2 = p_refl / (4 * np.pi * r_rx**2)
    # effective area of receive antenna:
    a_eff = G_rx * lambd**2 / (4 * np.pi)
    # received power:
    p_rx = s2 * a_eff
    return p_rx
