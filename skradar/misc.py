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
