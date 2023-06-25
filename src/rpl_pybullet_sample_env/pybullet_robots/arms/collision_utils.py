""" Collision Checking utils
Currently supports - Self Collision checking. Tested only on Panda
TODO: Environment Collision checking
TODO: Test on other robots

Credit to some of the functions to adamheins/pyb_utils
"""
import pybullet as p
import numpy as np
from dataclasses import dataclass

@dataclass
class NamedCollisionObject:
    """Name of a body and one of its links.
    The body name must correspond to the key in the `bodies` dict, but is
    otherwise arbitrary. The link name should match the URDF. The link name may
    also be None, in which case the base link (index -1) is used.
    """

    body_name: str
    link_name: str = None

@dataclass
class IndexedCollisionObject:
    """Index of a body and one of its links."""

    body_uid: int
    link_uid: int

def index_self_collision_pairs(physics_uid, robot_id, virtual_links=[]):
    """Convert a list of named collision pairs to indexed collision pairs.
    In other words, convert named bodies and links to the indexes used by
    PyBullet to facilate computing collisions between the objects.
    Parameters:
      physics_uid: Index of the PyBullet physics server to use.
      robot_id: Index of the Pybullet Robot
      virtual_links: Links that aren't physical - to skip collision checking
    Returns: a list of 2-tuples of IndexedCollisionObject
    """

    # build a nested dictionary mapping body names to link names to link
    # indices
    n = p.getNumJoints(robot_id, physics_uid)
    col_objects = []
    for i in range(n):
        col_objects.append(IndexedCollisionObject(robot_id, i))
    # Index every link pair (in both orders) of the robot
    indexed_collision_pairs = []
    for i, a_i in enumerate(col_objects):
        for j, b_j in enumerate(col_objects):
            # Skip adjacent links
            if i==j: continue
            if p.getJointInfo(robot_id, a_i.link_uid, physics_uid)[-1] == b_j.link_uid:
              continue
            if p.getJointInfo(robot_id, b_j.link_uid, physics_uid)[-1] == a_i.link_uid:
              continue
            # Skip imaginary links
            if a_i.link_uid in virtual_links or b_j.link_uid in virtual_links:
              continue
            indexed_collision_pairs.append((a_i, b_j))
    return indexed_collision_pairs



class CollisionDetector:
    """Collision Detector Class"""
    def __init__(self, physicsClient, robot, use_ghost, 
                named_collision_pairs=None, virtual_links=None):
        """
        :param physicsClient: PyBullet Physics Client
        :param robot: Robot object, Must have .ghost robot if use_ghost=True
        :param use_ghost: bool, whether to imagine the collision poses or take them
        :named_collision_pairs: Not Implemented yet.
        :param virtual_links: List, links within the robot that are not physical
        """
        self.physicsClient = physicsClient
        self.robot = robot
        if use_ghost:
            self.robot_id = robot.ghost.id
        else: 
            self.robot_id = robot.id
        self.use_ghost = use_ghost

        if named_collision_pairs is not None: 
            raise NotImplementedError("Environment Collision - Not included yet")
        if virtual_links is None: virtual_links=[]
        self.indexed_collision_pairs = index_self_collision_pairs(self.physicsClient, self.robot_id,
                                                                  virtual_links = virtual_links)

    def compute_distances(self, q, max_distance=1.0, debug=False):
        """Compute closest distances for a given configuration.
        Parameters:
          q: Iterable representing the desired configuration. This is applied
             directly to PyBullet body with index bodies["robot"].
          max_distance: Bodies farther apart than this distance are not queried
             by PyBullet, the return value for the distance between such bodies
             will be max_distance.
          debug: if True, prints the collision pair IDs
        Returns: A NumPy array of distances, one per pair of collision objects.
        """

        # put the robot in the given configuration
        if self.use_ghost:
            for i, q_i in enumerate(q):
                p.resetJointState(self.robot_id, i+1, q_i, physicsClientId=self.physicsClient)

        # compute shortest distances between all object pairs
        distances = []
        for a, b in self.indexed_collision_pairs:
            closest_points = p.getClosestPoints(
                a.body_uid,
                b.body_uid,
                distance=max_distance,
                linkIndexA=a.link_uid,
                linkIndexB=b.link_uid,
                physicsClientId=self.physicsClient,
            )

            # if bodies are above max_distance apart, nothing is returned, so
            # we just saturate at max_distance. Otherwise, take the minimum
            if len(closest_points) == 0:
                distances.append(max_distance)
            else:
                distances.append(np.min([pt[8] for pt in closest_points]))
                if debug and np.min([pt[8] for pt in closest_points])<max_distance:
                    print(a, b)
        return np.array(distances)

    def in_collision(self, q, margin=0, debug=False):
        """Returns True if configuration q is in collision, False otherwise.
        Parameters:
          q: Iterable representing the desired configuration.
          margin: Distance at which objects are considered in collision.
             Default is 0.0.
        """
        pair_ds = self.compute_distances(q, max_distance=margin, debug=debug)
        return (pair_ds < margin).any()
