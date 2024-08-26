import numpy as np


def nextpow2(N: float) -> int:
    """
    Returns the smallest integer x such that 2**x >= N

    Parameters
    ----------
    N : float
        Input value that should be expressed as a power of two using an integer
        as exponent.

    Returns
    -------
    x: int
        Smallest integer x such that 2**x >= N.

    """
    return int(np.ceil(np.log2(N)))

def dBm2W(val_dBm: np.ndarray) -> np.ndarray:
    """Converts values from dBm to Watt

    Parameters
    ----------
    val_dbm : float
        Input value(s) in dBm

    Returns
    -------
    float
        Output value(s) in Watt
    """
    return 10**(val_dBm/10)*1e-3

def dB2lin(val_dB: np.ndarray) -> np.ndarray:
    """Converts values from dB to linear

    Parameters
    ----------
    val_dB : float
        Input value(s) in dB

    Returns
    -------
    float
        Output value(s) in linear scale
    """
    return 10**(val_dB/10)