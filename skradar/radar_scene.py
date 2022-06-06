import pdb
import warnings
from abc import ABC, abstractmethod
from pytransform3d.transformations import vector_to_point, transform
from pytransform3d.transform_manager import TransformManager
import pytransform3d.transformations as pt
from pytransform3d.coordinates import spherical_from_cartesian
from scipy.constants import speed_of_light as c0
from skradar import range_compress_FMCW, sim_FMCW_if
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


class Thing:
    """
    A class describing position and orientation of a thing.

    Both location and orientation might change over time for non-zero
    velocities
    
    Attributes:
    -----------
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
                f'Warning: Expected pos with shape (3,1) but got {pos.shape}.'+
                ' Trying to reshape')
            pos = pos.copy().reshape((3, 1))
        self.pos = pos
        if not(vel.shape == (3, 1)):
            warnings.warn(
                f'Warning: Expected vel with shape (3,1) but got {vel.shape}.'+
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
            R=self.rotation, p=self.pos.ravel())  # homogeneous coordinates
        world2loc = pt.invert_transform(self.loc2world)
        return pos, loc2world, world2loc

    def update_pose(self, delta_t: float):
        pos, loc2world, world2loc = self.predict_pose(delta_t)
        self.pos = pos
        self.loc2world = loc2world
        self.world2loc = world2loc


class Target(Thing):
    """
    A class describing a radar target (derived from Thing).

    A target has a radar cross section (in addition to the position and
    orientation information)
    
    Attributes:
    -----------
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
        ----------
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
    
    Attributes:
    -----------
        tx_pos : np.ndarray, shape(3, M_tx)
            Local 3-d Cartesian coordinate(s) of the M_tx TX antenna(s) in meters.
        rx_pos : np.ndarray, shape(3, M_rx)
            Local 3-d Cartesian coordinate(s) of the M_rx RX antenna(s) in meters.
    """

    def __init__(self, tx_pos: np.ndarray, rx_pos: np.ndarray, **kwargs):
        """
        Create a radar object.

        Parameters
        ----------
        tx_pos : np.ndarray, shape(3, M_tx)
            Local 3-d Cartesian coordinate(s) of the M_tx TX antenna(s) in meters.
        rx_pos : np.ndarray, shape(3, M_rx)
            Local 3-d Cartesian coordinate(s) of the M_rx RX antenna(s) in meters.
        **kwargs : optional
            Remaining keyword arguments are passed to constructor of the
            Thing class.

        Returns
        -------
        None.

        """
        self.tx_pos = tx_pos
        self.rx_pos = rx_pos
        self.targets = None
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

    def set_targets(self, targets: list[Target]):
        self.targets = targets

    def calc_dists(self, delta_t: float = 0) -> np.ndarray:
        """
        Calculate the distances between all target(s) and TX as well as RX
        antennas. The optional time parameter delta_t can be used to
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
                          targ_cntr] = tx_dist[tx_cntr]+rx_dist[rx_cntr]
        return dists

    @abstractmethod
    def range_compression(self):
        """
        Needs to be implemented in subclass.

        Returns
        -------
        None.

        """
        pass

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
    
    Attributes:
    -----------
        TODO:
            TODO.
    """

    def __init__(self, B: float, fc: float, N_f: int, N_s: int, T_f: float,
                 T_s: float, **kwargs):
        self.B = B
        self.fc = fc
        self.N_f = N_f
        self.N_s = N_s
        self.T_f = T_f
        self.T_s = T_s
        self.s_if = None
        super().__init__(**kwargs)
        self.t_s = np.arange(N_s)*T_s  # slow time vec. (unif. chirp sequence)

    def sim_chirps(self):
        self.s_if = np.zeros((self.M_tx, self.M_rx, self.N_s, self.N_f))
        for chirp_cntr in range(self.N_s):
            dists = self.calc_dists(chirp_cntr*self.T_s)
            for tx_cntr in range(self.M_tx):
                for rx_cntr in range(self.M_rx):
                    for targ_cntr in range(self.N_targ):  # sum over targets
                        dist = dists[tx_cntr, rx_cntr, targ_cntr]
                        s_if_tmp = sim_FMCW_if(
                            dist, self.B, self.fc, self.N_f, self.T_s)
                        self.s_if[tx_cntr, rx_cntr, chirp_cntr, :] += s_if_tmp

    def range_compression(self, zp_fact: int):
        if self.s_if is None:
            raise TypeError(
                's_if is None. It has to be simulated or loaded first')
        else:
            flatten_phase = True
            self.rp, self.ranges = range_compress_FMCW(self.s_if, self.B, zp_fact,
                                          self.scene.c, flatten_phase)


class Scene:
    """
    A class describing a radar scene.

    The scene can include several radars and targets at different positions and
    with different orientations.
    
    Attributes:
    -----------
        radars: list
            A list of Radar objects.
        targets: list
            A list of Target objects.
    """

    def __init__(self, radars: list, targets: list, c: float=c0):
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

    def visualize(self, frame, ax):
        scaling = 0.1
        self.tm.plot_frames_in(frame, ax=ax, s=scaling)
        ax.set_xlim((-0.25, 0.75))
        ax.set_ylim((-0.5, 0.5))
        ax.set_zlim((0.0, 1.0))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


if __name__ == "__main__":
    B = 1e9
    fc = 76.5e9
    N_f = 1024  # number of fast time samples
    f_sf = 1e6  # fast time sampling rate
    N_s = 1  # number of slow time samples
    T_chirp = (N_f-1)*1/f_sf

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
                       [139.425, -20.25, 16.0]]).T*1e-3

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
                       [54.6, 20.25, 16.0]]).T*1e-3

    radar = FMCWRadar(B=B, fc=fc, N_f=N_f, T_f=1/f_sf, T_s=1/T_chirp,
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
    plt.plot(radar.ranges/2, 20*np.log10(np.abs(radar.rp[0,0,0,:])))
