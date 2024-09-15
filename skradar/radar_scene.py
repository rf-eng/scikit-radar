import pdb
import warnings
from abc import ABC, abstractmethod
from pytransform3d.transformations import vector_to_point, transform
from pytransform3d.transform_manager import TransformManager
import pytransform3d.transformations as pt
from pytransform3d.coordinates import spherical_from_cartesian
from scipy.constants import speed_of_light as c0, Boltzmann
from skradar import range_compress_FMCW, sim_FMCW_if, backprojection
from skradar import nextpow2, dBm2W, dB2lin
import numpy as np
import scipy.signal
import scipy.spatial
import matplotlib.pyplot as plt
plt.ion()


class Thing:
    """
    A class describing position and orientation of a thing.

    Both location and orientation might change over time for non-zero
    velocities

    Attributes
    ----------
        pos:
            Vector describing the current position in Cartesian
            world coordinates.
        rotation:  
            Rotation matrix describing the orientation of the object.
        vel:  
            Current velocity vector.
        omega:  
            Current angular velocity vector.
    """

    def __init__(self, name: str, pos: np.ndarray,
                 rotation: np.ndarray = np.eye(3),
                 vel: np.ndarray = np.zeros((3, 1))):
        """
        Create a thing.

        Parameters
        ----------
        name : str
            Name of the object to simplify its identification for humans.
        pos : np.ndarray, shape (3, 1)
            Position of the object (=origin of local coordinates)
            in world coordinates.
        rotation : np.ndarray, shape (3, 3)
            Rotation matrix describing the orientation of the object.
            The default is np.eye(3).
        vel : np.ndarray, shape (3, 1)
            Velocity vector in world coordinates. The default is np.zeros(3,1).
        """
        if not(pos.shape == (3, 1)):
            warnings.warn(
                f'Warning: Expected pos with shape (3,1) but got {pos.shape}.' +
                ' Trying to reshape')
            pos = pos.copy().reshape((3, 1))
        self.pos = pos
        if not(vel.shape == (3, 1)):
            warnings.warn(
                f'Warning: Expected vel with shape (3,1) but got {vel.shape}.' +
                ' Trying to reshape')
            vel = vel.copy().reshape((3, 1))
        self.vel = vel
        self.rotation = rotation
        self.loc2world = pt.transform_from(
            R=self.rotation, p=pos.ravel())  # homogeneous coordinates
        self.world2loc = pt.invert_transform(self.loc2world)
        self.name = name
        self.scene = None

    def __str__(self):  # convenient for visualization
        if not(self.name is None) and len(self.name) > 0:
            return self.name
        else:
            return self.__repr__()

    def predict_pose(self, delta_t: float) -> tuple[np.ndarray, np.ndarray,
                                                    np.ndarray]:
        pos = self.pos + self.vel * delta_t
        loc2world = pt.transform_from(
            R=self.rotation, p=pos.ravel())  # homogeneous coordinates
        world2loc = pt.invert_transform(loc2world)
        return pos, loc2world, world2loc

    def update_pose(self, delta_t: float):
        pos, loc2world, world2loc = self.predict_pose(delta_t)
        self.pos = pos
        self.loc2world = loc2world
        self.world2loc = world2loc
        self.scene.tm.add_transform("world", self, self.world2loc)


class Target(Thing):
    """
    A class describing a radar target (derived from Thing).

    A target has a radar cross section (in addition to the position and
    orientation information)

    Attributes
    ----------
        rcs:
            Radar cross section in square meters.
    """

    def __init__(self, rcs: float, **kwargs):
        """
        Create a target.

        Parameters
        ----------
        rcs : float
            Radar cross section in square meters.

        Other Parameters
        ----------------
        **kwargs : optional
            Remaining keyword arguments are passed to constructor of the
            Thing class.
        """
        super().__init__(**kwargs)
        self.rcs = rcs


class Radar(Thing, ABC):
    """
    A class describing a generic radar (derived from Thing).

    The generic radar only allows to determine relative ranges and angles
    between the radar antennas and (several) target(s)

    Attributes
    ----------
        tx_pos : np.ndarray, shape(3, M_tx)
            Local 3-d Cartesian coordinate(s) of the M_tx TX antenna(s) in meters.
        rx_pos : np.ndarray, shape(3, M_rx)
            Local 3-d Cartesian coordinate(s) of the M_rx RX antenna(s) in meters.
    """

    def __init__(self, tx_pos: np.ndarray, rx_pos: np.ndarray, N_s: int, T_s: float,
                 tx_ant_gains: np.ndarray = None, rx_ant_gains: np.ndarray = None,
                 tx_powers: np.ndarray = None, rx_gains: np.ndarray = None,
                 Z0: float = 50, T_ref: float = 293,
                 **kwargs):
        """
        Create a radar object.

        Parameters
        ----------
        tx_pos : np.ndarray, shape(3, M_tx)
            Local 3-d Cartesian coordinate(s) of the M_tx TX antenna(s) in meters.
        rx_pos : np.ndarray, shape(3, M_rx)
            Local 3-d Cartesian coordinate(s) of the M_rx RX antenna(s) in meters.
        N_s : int
            Number of slow time samples.
        T_s : float
            Slow time sampling interval in seconds.
        tx_ant_gains : np.ndarray, shape(M_tx), optional
            Antenna gains of TX antenna(s) in dBi. Default: 0 dBi.
        rx_ant_gains : np.ndarray, shape(M_rx), optional
            Antenna gains of RX antenna(s) in dBi. Default: 0 dBi.
        tx_powers : np.ndarray, shape(M_tx), optional
            Transmit powers at the inputs of each TX antenna in dBm. Default: 0 dBm.
        rx_gains : np.ndarray, shape(M_rx), optional
            Gain(s) in the RX sections(s) in dB. Default: 0 dB.
        Z0 : float, optional
            Reference impedance in Ohm. Default: 50 Ohm.
        T_ref : float, optional
            Reference temperature in Kelvin. Default: Room temperature 293 K.
        **kwargs : optional
            Remaining keyword arguments are passed to constructor of the
            Thing class.

        Returns
        -------
        None.

        """
        if tx_pos.ndim != 2:
            raise ValueError(
                f'Expected two-dimensional tx_pos but got {tx_pos.ndim}.')
        elif tx_pos.shape[0] != 3:
            raise ValueError(
                f'Expected tx_pos with 3 rows but got {tx_pos.shape[0]}.')
        else:
            self.tx_pos = tx_pos
        if rx_pos.ndim != 2:
            raise ValueError(
                f'Expected two-dimensional rx_pos but got {rx_pos.ndim}.')
        elif rx_pos.shape[0] != 3:
            raise ValueError(
                f'Expected rx_pos with 3 rows but got {rx_pos.shape[0]}.')
        else:
            self.rx_pos = rx_pos
        if tx_ant_gains is None:
            self.tx_ant_gains = np.ones(self.M_tx)  # isotropic radiator
        else:
            self.tx_ant_gains = dB2lin(tx_ant_gains)
        if rx_ant_gains is None:
            self.rx_ant_gains = np.ones(self.M_rx)  # isotropic radiator
        else:
            self.rx_ant_gains = dB2lin(rx_ant_gains)
        if tx_powers is None:
            self.tx_powers = dBm2W(np.zeros(self.M_tx))  # zero dBm on all TXs
        else:
            self.tx_ant_gains = dBm2W(tx_powers)
        if rx_gains is None:
            self.rx_gains = np.ones(self.M_rx)  # 0 dB on all RXs
        else:
            self.rx_gains = dB2lin(rx_gains)
        self.Z0 = Z0
        self.T_ref = T_ref
        self.targets = None
        self.rp = None  # range profile
        
        # slow time vec. (unif. chirp sequence)
        self.t_s = np.arange(N_s) * T_s

        # random number generator
        self.rng = np.random.default_rng()
        super().__init__(**kwargs)

    @property
    def M_tx(self):
        return self.tx_pos.shape[1]

    @property
    def M_rx(self):
        return self.rx_pos.shape[1]

    @property
    def N_targ(self):
        return len(self.targets)

    @property
    def kw(self):
        return 2 * np.pi * self.fc / self.scene.c

    def set_targets(self, targets: list[Target]):
        self.targets = targets

    def calc_dists(self, delta_t: float = 0) -> np.ndarray:
        """
        Calculate the distances between all target(s) and TX as well as RX antennas.

        The optional time parameter delta_t can be used to
        calculate varying distances within a CPI. It is assumed that target and
        radar velocities are constant over the CPI.

        Parameters
        ----------
        delta_t : float, optional
            Prediction time in seconds. Can, for example, be used to calculate
            varying distances over a CPI.

        Returns
        -------
        dists : np.ndarray, shape(M_tx, M_rx, N_targ)
            Array containing all distances between TXs, RXs, and targets.

        """
        # np.sqrt(np.dot(vec, vec)) is faster than np.linalg.norm
        # TODO: consider using scipy.spatial.distance_matrix
        dists = np.empty((self.M_tx, self.M_rx, self.N_targ))
        tx_dist = np.empty(self.M_tx)
        rx_dist = np.empty(self.M_rx)
        for targ_cntr, target in enumerate(self.targets):
            # convert target position vector to homogeneous coordinates:
            targ_pos = target.predict_pose(delta_t)[0]
            p_in_world = vector_to_point(targ_pos.ravel())
            # calculate vector from local origin to target
            world2loc = self.predict_pose(delta_t)[2]
            p_in_radar = transform(world2loc, p_in_world)
            for tx_cntr in range(self.M_tx):  # calc. all distances tx <-> targ
                tx_to_targ = p_in_radar[0:3] - self.tx_pos[:, tx_cntr]
                tx_dist[tx_cntr] = np.sqrt(np.dot(tx_to_targ, tx_to_targ))
            for rx_cntr in range(self.M_rx):  # calc. all distances rx <-> targ
                rx_to_targ = p_in_radar[0:3] - self.rx_pos[:, rx_cntr]
                rx_dist[rx_cntr] = np.sqrt(np.dot(rx_to_targ, rx_to_targ))
            for tx_cntr in range(self.M_tx):
                for rx_cntr in range(self.M_rx):  # all tx/rx combinations
                    dists[tx_cntr, rx_cntr,
                          targ_cntr] = tx_dist[tx_cntr] + rx_dist[rx_cntr]
        return dists, tx_dist, rx_dist

    @abstractmethod
    def range_compression(self):
        """
        Needs to be implemented in subclass since it depends on the type of radar modulation.

        Returns
        -------
        None.

        """
        pass

    def merge_mimo(self):
        raise NotImplementedError('Function not implemented yet')

    def extract_mimo(self):
        raise NotImplementedError('Function not implemented yet')

    def angle_proc_bp(self, ranges_bp: np.ndarray, angles_bp: np.ndarray,
                      process_noisy: bool = True):
        # ranges_bp contains distances to image pixels (not round-trip)
        if self.rp is None:
            raise TypeError(
                'rp is None. It has to be calculated first')
        elif self.M_rx < 2 and self.M_tx < 2:
            warnings.warn("Warning: Angle processing doesn't make sense for " +
                          "only one antenna. Doing nothing.")
        elif 2 * (np.max(ranges_bp)) > np.max(self.ranges):
            # Definitely too large, even for a monostatic configuration
            raise ValueError('Image size too large: Range profile does not ' +
                             'cover the largest distance TX->pixel->RX.')
        else:
            self.ranges_bp = ranges_bp
            num_ranges = self.ranges_bp.shape[0]
            self.angles_bp = angles_bp
            num_angles = self.angles_bp.shape[0]
            # all range-angle combinations:
            r_mat, ang_mat = np.meshgrid(self.ranges_bp, self.angles_bp)
            x_mat = r_mat * np.sin(ang_mat)
            y_mat = np.zeros_like(x_mat)
            z_mat = r_mat * np.cos(ang_mat)

            # all combinations of TX-, RX-, and pixel-indices
            rx_idcs_mat, tx_idcs_mat, px_idcs_mat = np.meshgrid(
                np.arange(self.M_rx), np.arange(self.M_tx), np.arange(num_ranges * num_angles))

            tx_idcs = tx_idcs_mat.ravel()
            rx_idcs = rx_idcs_mat.ravel()
            px_idcs = px_idcs_mat.ravel()

            image_vec = backprojection(x_mat, y_mat, z_mat, (self.tx_pos, self.rx_pos),
                           (tx_idcs, rx_idcs), px_idcs, self.rp, self.ranges, self.kw, self.N_f)

            self.ra_bp = image_vec.reshape((num_angles, num_ranges))
            scale_to_amp = 1 / (self.M_rx * self.M_tx)
            self.ra_bp = scale_to_amp * self.ra_bp
            if process_noisy:
                image_vec = backprojection(x_mat, y_mat, z_mat, (self.tx_pos, self.rx_pos),
                           (tx_idcs, rx_idcs), px_idcs, self.rp_noisy, self.ranges, self.kw, self.N_f)
                self.ra_bp_noisy = image_vec.reshape((num_angles, num_ranges))
                self.ra_bp_noisy = scale_to_amp * self.ra_bp_noisy

    def angle_proc_RX_DFT(self, zp_fact: float = 1, win_rx: str = 'boxcar',
                          process_noisy: bool = True):
        """
        Calculate angle-FFT along the RX antennas for all range-profile samples.

        Parameters
        ----------
        zp_fact : float, optional
            Zero-padding factor
        win_rx : str, optional
            Window function used for RX-only beamforming. Default is 'boxcar'.
        process_noisy : bool
            Determines if both noiseless and noisy signals are processed. Default: true

        Returns
        -------
        None.

        """
        if self.rp is None:
            raise TypeError(
                'rp is None. It has to be calculated first')
        elif self.M_rx < 2:
            warnings.warn("Warning: Angle processing doesn't make sense for " +
                          "only one antenna. Doing nothing.")
        else:
            # TODO: Check if antennas are on a line
            # TODO: Check if spacing is uniform
            win_rx = scipy.signal.windows.get_window(win_rx, self.M_rx)
            win = win_rx[:, np.newaxis, np.newaxis]
            z = 2**nextpow2(zp_fact * self.M_rx)
            win_coh_gain = np.sum(win_rx) / self.M_rx
            scale_to_amp = 1 / (self.M_rx * win_coh_gain)
            self.ra = scale_to_amp * np.fft.fft(self.rp * win, n=z, axis=-3)
            self.ra = np.fft.fftshift(self.ra, axes=-3)
            if process_noisy:
                self.ra_noisy = scale_to_amp * np.fft.fft(self.rp_noisy * win, n=z, axis=-3)
                self.ra_noisy = np.fft.fftshift(self.ra_noisy, axes=-3)
            u_vec = 2 * np.fft.fftshift(np.fft.fftfreq(z))
            self.angles = np.arcsin(u_vec)

    def angle_proc_DFT(self):
        self.extract_mimo()
        raise NotImplementedError('Function not implemented yet')

    def doppler_processing(self, zp_fact: float = 1, win_doppler: str = "boxcar",
                           process_noisy: bool=True):
        """
        Calculate range-Doppler map by applying an FFT along the slow time.

        Parameters
        ----------
        zp_fact : float, optional
            Zero-padding factor, by default 1
        win_doppler : str, optional
            Window function used for Doppler processing. Default is 'boxcar'.
        process_noisy : bool
            Determines if both noiseless and noisy signals are processed. Default: true

        """
        if self.rp is None:
            raise TypeError(
                'rp is None. It has to be calculated first')
        else:
            if self.N_s > 1:
                win = scipy.signal.windows.get_window(win_doppler, self.N_s)
                win = win[np.newaxis, np.newaxis, :, np.newaxis]
                z = 2**nextpow2(zp_fact * self.N_s)
                self.rd = np.fft.fft(self.rp, n=z, axis=2)
                if process_noisy:
                    if self.rp_noisy is None:
                        raise TypeError(
                            'rp_noisy is None. It has to be calculated first')
                    self.rd_noisy = np.fft.fft(self.rp_noisy, n=z, axis=2)
            else:
                self.rd = self.rp

    def process_radar_cube(self):
        self.range_compression(self)
        self.doppler_processing(self)
        self.mimo_processing(self)
        self.angle_processing(self)


class FMCWRadar(Radar):
    """
    A class describing an FMCW radar (derived from Radar).

    The FMCW radar has to be configured with some radar settings and allows
    to simulate actual radar signals.

    Attributes
    ----------
        TODO:
            TODO.
    """

    def __init__(self, B: float, fc: float, N_f: int, N_s: int, T_f: float,
                 T_s: float, win_range: str = 'hann', win_doppler: str = "boxcar",
                 if_real: bool = True, **kwargs):
        self.B = B
        self.fc = fc
        self.N_f = N_f
        self.N_s = N_s
        self.T_f = T_f
        self.T_s = T_s
        self.if_real = if_real
        self.s_if = None
        self.s_if_bursts = None
        self.win_range = scipy.signal.windows.get_window(win_range, N_f)
        self.win_doppler = scipy.signal.windows.get_window(win_doppler, N_s)
        super().__init__(N_s=N_s, T_s=T_s, **kwargs)

    def sim_chirps(self):
        """
        Simulate IF signals (slow-time and fast-time samples) for the FMCW radar.
        
        The simulated signal results from the sum of the contributions from all 
        TX, RX and target combinations.

        Returns
        -------
        None.

        """
        if self.if_real:
            self.s_if = np.zeros((self.M_tx, self.M_rx, self.N_s, self.N_f))
            self.s_if_noisy = np.zeros((self.M_tx, self.M_rx, self.N_s, self.N_f))
        else:
            self.s_if = np.zeros((self.M_tx, self.M_rx, self.N_s, self.N_f), dtype=complex)
            self.s_if_noisy = np.zeros((self.M_tx, self.M_rx, self.N_s, self.N_f), dtype=complex)
        for chirp_cntr in range(self.N_s):
            dists, tx_dist, rx_dist = self.calc_dists(chirp_cntr * self.T_s)
            for tx_cntr in range(self.M_tx):
                for rx_cntr in range(self.M_rx):
                    for targ_cntr in range(self.N_targ):  # sum over targets
                        dist = dists[tx_cntr, rx_cntr, targ_cntr]
                        # radar equation step-by-step
                        p_tx_eirp = self.tx_powers[tx_cntr] * self.tx_ant_gains[tx_cntr]
                        s_targ = p_tx_eirp/(4*np.pi*tx_dist[tx_cntr]**2)  # power density at target
                        p_refl = s_targ*self.targets[targ_cntr].rcs  # power reflected from target
                        s_rx = p_refl/(4*np.pi*rx_dist[rx_cntr]**2)  # power density at receiver
                        lambd = c0/self.fc
                        a_eff = self.rx_ant_gains[rx_cntr]*lambd**2/(4*np.pi)  # effective RX antenna area
                        p_rx = s_rx * a_eff * self.rx_gains[rx_cntr]  # power of IF signal
                        A_pk = np.sqrt(2)*np.sqrt(p_rx * self.Z0)  # power to amplitude
                        s_if_tmp = A_pk*sim_FMCW_if(
                            dist, self.B, self.fc, self.N_f, self.T_s, cplx=not(self.if_real))
                        self.s_if[tx_cntr, rx_cntr, chirp_cntr, :] += s_if_tmp
        self.generate_AWGN()

    def generate_AWGN(self):
        fs = 1/self.T_f
        noise_std = np.sqrt(4*self.Z0*Boltzmann*self.T_ref*fs/2)
        if self.if_real:
            self.noise = self.rng.normal(0, noise_std, (self.M_rx, self.N_s, self.N_f))
        else:
            self.noise = (self.rng.normal(0, noise_std/np.sqrt(2), (self.M_rx, self.N_s, self.N_f))+
                         1j*self.rng.normal(0, noise_std/np.sqrt(2), (self.M_rx, self.N_s, self.N_f)))
        # multiple TXs and/or multiple targets do not add noise
        # noise will be only added for each RX, slow-time sample and fast-time sample
        tx_cntr = 0
        for chirp_cntr in range(self.N_s):
            for rx_cntr in range(self.M_rx):
                self.s_if_noisy[tx_cntr, rx_cntr,
                                chirp_cntr, :] = (self.s_if[tx_cntr, rx_cntr, chirp_cntr, :] + 
                                                    self.noise[rx_cntr, chirp_cntr, :])
                        
    def add_burst(self):
        """
        Generate a new burst of IF signals and append to the array of bursts.

        Returns
        -------
        None.

        """
        self.sim_chirps(self)
        if self.s_if_bursts is None:
            self.s_if_bursts = self.s_if[:, :, np.newaxis, :, :]
        else:
            np.concatenate(
                (self.s_if_bursts, self.s_if[:, :, np.newaxis, :, :]), axis=2)

    def range_compression(self, zp_fact: float, process_noisy: bool=True):
        """
        Perform range compression and amplitude scaling on the previously simulated or measured intermediate frequency data.

        The amplitude scaling takes the number of samples and the coherent
        window gain into account.

        Parameters
        ----------
        zp_fact : float            
            Zero-padding factor. The IF signal is zero-padded to
            2**nextpow2(zp_fact*N) with N being the number of samples in s_if.
        process_noisy : bool
            Determines if both noiseless and noisy signals are processed. Default: true

        Raises
        ------
        TypeError
            A TypeError is raised if no s_if is available from a previous
            simulation or measurement.

        Returns
        -------
        None.

        """
        if self.s_if is None:
            raise TypeError(
                's_if is None. It has to be simulated or loaded first')
        else:
            flatten_phase = True
            win_coh_gain = np.sum(self.win_range) / self.N_f
            if self.if_real:
                scale_to_amp = 2 / (self.N_f * win_coh_gain)
            else:
                scale_to_amp = 1 / (self.N_f * win_coh_gain)
            self.rp, self.ranges = range_compress_FMCW(
                self.s_if, self.win_range, self.B, zp_fact,
                self.scene.c, flatten_phase)
            self.rp = scale_to_amp * self.rp
            if process_noisy:
                if self.s_if_noisy is None:
                    raise TypeError(
                        's_if is None. It has to be simulated or loaded first')
                self.rp_noisy, self.ranges = range_compress_FMCW(
                    self.s_if_noisy, self.win_range, self.B, zp_fact,
                    self.scene.c, flatten_phase)
                self.rp_noisy = scale_to_amp * self.rp_noisy


class Scene:
    """
    A class describing a radar scene.

    The scene can include several radars and targets at different positions and
    with different orientations.

    Attributes
    ----------
        radars: list
            A list of Radar objects.
        targets: list
            A list of Target objects.
    """

    def __init__(self, radars: list, targets: list, c: float = c0):
        """


        Parameters
        ----------
        radars : list
            List of radar objects.
        targets : list
            List of target objects.
        c : float, optional
            Propagation velocity in meters/second. Default is c0

        Returns
        -------
        None.

        """

        self.radars = radars
        self.targets = targets
        self.tm = TransformManager()
        self.c = c
        for radar in self.radars:
            self.tm.add_transform("world", radar, radar.world2loc)
            radar.set_targets(targets)
            radar.scene = self
        for target in self.targets:
            self.tm.add_transform("world", target, target.world2loc)
            target.scene = self

    def update_poses(self, delta_t: float):
        """
        Use current velocities and angular velocities to calculate and
        set new poses for delta_t after the current time.

        Parameters
        ----------
        delta_t : float
            Time step in seconds.
        """
        for radar in self.radars:
            radar.update_pose(delta_t)
        for target in self.targets:
            target.update_pose(delta_t)

    def visualize(self, frame, ax, coord_len: float = 1):
        self.tm.plot_frames_in(frame, ax=ax, s=coord_len)


if __name__ == "__main__":
    B = 1e9
    fc = 76.5e9
    N_f = 128  # number of fast time samples
    f_sf = 1e6  # fast time sampling rate
    N_s = 1  # number of slow time samples
    T_chirp = (N_f - 1) * 1 / f_sf

    reference_pos = np.array([[0], [0], [0.3]])
    TxPosn = np.array([[-139.425, -20.25, 16.0],
                       [-129.675, -20.25, 16.0],
                       [-119.925, -20.25, 16.0],
                       [-110.175, -20.25, 16.0],
                       [-14.625, -20.25, 16.0],
                       [-4.875, -20.25, 16.0],
                       [4.875, -20.25, 16.0],
                       [14.625, -20.25, 16.0],
                       [110.175, -20.25, 16.0],
                       [119.925, -20.25, 16.0],
                       [129.675, -20.25, 16.0],
                       [139.425, -20.25, 16.0]]).T * 1e-3

    RxPosn = np.array([[-62.4, 20.25, 16.0],
                       [-54.6, 20.25, 16.0],
                       [-46.8, 20.25, 16.0],
                       [-39.0, 20.25, 16.0],
                       [-31.2, 20.25, 16.0],
                       [-23.4, 20.25, 16.0],
                       [-15.6, 20.25, 16.0],
                       [-7.8, 20.25, 16.0],
                       [0.0, 20.25, 16.0],
                       [7.8, 20.25, 16.0],
                       [15.6, 20.25, 16.0],
                       [23.4, 20.25, 16.0],
                       [31.2, 20.25, 16.0],
                       [39.0, 20.25, 16.0],
                       [46.8, 20.25, 16.0],
                       [54.6, 20.25, 16.0]]).T * 1e-3

    radar = FMCWRadar(B=B, fc=fc, N_f=N_f, T_f=1 / f_sf, T_s=1 / T_chirp,
                      N_s=N_s, tx_pos=TxPosn, rx_pos=RxPosn,
                      pos=reference_pos, name='First radar')
    target1 = Target(rcs=1, pos=np.array(
        [[0.2], [0.5], [0.5]]), name='Small static target')
    target2 = Target(rcs=10, pos=np.array(
        [[0], [0], [0.7]]), name='Big static target')
    scene = Scene([radar], [target1, target2])

    # Visualize scene
    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    scene.visualize("world", ax)

    # Print some geometric information
    tf_matrix = scene.tm.get_transform(radar, target1)
    print(f'Transformation matrix between radar and target1: {tf_matrix}\n')
    p_in_world = vector_to_point(np.array([0, 0, 0.5]))
    p_in_radar = transform(radar.world2loc, p_in_world)
    p_in_target1 = transform(target1.world2loc, p_in_world)

    print(f'The point {p_in_world} in the world coordinate can be described as'
          f' {p_in_radar} in the coordinate system of the radar and as'
          f' {p_in_target1} in the coordinate system of target1.')

    radar.sim_chirps()
    radar.range_compression(zp_fact=4)

    plt.figure(2)
    plt.clf()
    plt.plot(radar.ranges / 2, 20 * np.log10(np.abs(radar.rp[0, 0, 0, :])))

    delta_r = radar.ranges[1] - radar.ranges[0]
    ranges_bp = np.arange(start=0, stop=5, step=delta_r)
    radar.angle_proc_bp(ranges_bp, np.linspace(-np.pi / 2, np.pi / 2, 20))
