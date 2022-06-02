from pytransform3d.transformations import vector_to_point, transform
from pytransform3d.transform_manager import TransformManager
import pytransform3d.transformations as pt
from pytransform3d.coordinates import spherical_from_cartesian
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
                 vel: np.ndarray = np.zeros((3, 1)),
                 omega: np.ndarray = np.zeros((3, 1))):
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
            Initial velocity vector. The default is np.zeros(3,1).
        omega : np.ndarray, shape (3, 1)
            Initial angular velocity vector. The default is np.zeros(3,1).
        """
        self.pos = pos
        self.loc2world = pt.transform_from(
            R=rotation, p=pos.ravel())  # homogeneous coordinates
        self.world2loc = pt.invert_transform(self.loc2world)
        self.name = name

    def __str__(self):
        if not(self.name is None) and len(self.name) > 0:
            return self.name
        else:
            return self.__repr__()


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


class Radar(Thing):
    """
    A class describing a generic radar (derived from Thing).

    The generic radar only allows to determine relative ranges and angles
    between the radar and (several) target(s)
    
    Attributes:
    -----------
        TODO:
            TODO.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calc_target_locations(self, targets: list):
        for target in targets:
            p_in_world = vector_to_point(target.pos)  # to homogeneous coords.
            p_in_radar = transform(self.world2loc, p_in_world)
            print(spherical_from_cartesian(p_in_radar[0:3]))
    

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
    def __init__(self, B: float, fc: float, N: int, T_s: float):        
        self.B = B
        self.fc = fc
        self.N = N
        self.T_s = T_s


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

    def __init__(self, radars: list, targets: list):
        """
        

        Parameters
        ----------
        radars : list
            List of radar objects.
        targets : list
            List of target objects.

        Returns
        -------
        None.

        """

        self.radars = radars
        self.targets = targets
        self.tm = TransformManager()
        for radar in self.radars:
            self.tm.add_transform("world", radar, radar.world2loc)
        for target in self.targets:
            self.tm.add_transform("world", target, target.world2loc)

    def visualize(self, frame, ax):
        scaling = 0.1
        self.tm.plot_frames_in(frame, ax=ax, s=scaling)
        ax.set_xlim((-0.25, 0.75))
        ax.set_ylim((-0.5, 0.5))
        ax.set_zlim((0.0, 1.0))
        plt.show()


if __name__ == "__main__":
    B = 1e9
    fc = 76.5e9
    N = 1024
    f_s = 1e6
    radar = Radar(pos=np.array([0, 0, 0.3]), name='First radar')
    target1 = Target(rcs=1, pos=np.array(
        [0.2, 0.5, 0.5]), name='Small static target')
    target2 = Target(rcs=10, pos=np.array(
        [0, 0, 0.7]), name='Big static target')
    scene = Scene([radar], [target1, target2])
    
    radar.calc_target_locations([target1, target2])

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

    print(f'The point {p_in_world} in the world coordinate can be described as'\
          f' {p_in_radar} in the coordinate system of the radar and as'\
          f' {p_in_target1} in the coordinate system of target1.')
