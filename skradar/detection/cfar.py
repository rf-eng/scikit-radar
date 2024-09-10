import dataclasses
from enum import Enum, auto

import numpy as np


class CFARMode(Enum):
    CA = auto()
    CAGO = auto()


@dataclasses.dataclass(frozen=True)
class CFARConfig:
    mode: CFARMode = CFARMode.CA
    train_cells: int = 3
    guard_cells: int = 0
    pfa: float = 1e-4


DEFAULT_CFG = CFARConfig()


def cfar_threshold(sig: np.ndarray, cfg: CFARConfig = DEFAULT_CFG):
    """Calculates the threshold level for a signal x with a CFAR method given by mode, by looping over all cells.

    Args:
        sig: array of positive (absolute values) of floats
        cfg: CFAR configuration

    Returns:
        array of size of x holding threshold levels
    """
    scale = 1 / cfg.train_cells

    kernel = np.full(cfg.train_cells, scale)
    pad = cfg.train_cells + cfg.guard_cells
    sig_pad = np.hstack((sig[-pad:], sig, sig[:pad]))
    corr = np.correlate(sig_pad, kernel, mode='valid')

    offset = cfg.train_cells + 2 * cfg.guard_cells + 1
    left = corr[:-offset]
    right = corr[offset:]

    match cfg.mode:
        case CFARMode.CA:
            corr = 0.5 * (left + right)
        case CFARMode.CAGO:
            corr = np.maximum(left, right)
        case _:
            raise ValueError(f"CFAR mode '{str(cfg.mode)}' is not supported")

    L = 2 * cfg.train_cells
    alpha = np.sqrt(4 / np.pi * L * (cfg.pfa**(-1 / L) - 1) *
                    (1 - (1 - np.pi / 4) * np.exp(-L + 1)))  # see doi.org/10.1049/el:19891131

    threshold = alpha * corr
    return threshold
