from typing import Union, Optional, Dict, Any

import numpy as np


def distance(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    """Compute the distance between two array. This function is vectorized.
    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
    Returns:
        Union[float, np.ndarray]: The distance between the arrays.
    """
    assert a.shape == b.shape
    return np.linalg.norm(a - b, axis=-1)


def angle_distance(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    """Compute the geodesic distance between two array of angles. This function is vectorized.
    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
    Returns:
        Union[float, np.ndarray]: The geodesic distance between the angles.
    """
    assert a.shape == b.shape
    dist = 1 - np.inner(a, b) ** 2
    return dict



def create_geometry(
    self,
    body_name: str,
    geom_type: int,
    mass: float = 0.0,
    position: Optional[np.ndarray] = None,
    ghost: bool = False,
    lateral_friction: Optional[float] = None,
    spinning_friction: Optional[float] = None,
    visual_kwargs: Dict[str, Any] = {},
    collision_kwargs: Dict[str, Any] = {},
) -> None:
    """Create a geometry.
    Args:
        body_name (str): The name of the body. Must be unique in the sim.
        geom_type (int): The geometry type. See self.physics_client.GEOM_<shape>.
        mass (float, optional): The mass in kg. Defaults to 0.
        position (np.ndarray, optional): The position, as (x, y, z). Defaults to [0, 0, 0].
        ghost (bool, optional): Whether the body can collide. Defaults to False.
        lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
            value. Defaults to None.
        spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
            value. Defaults to None.
        visual_kwargs (dict, optional): Visual kwargs. Defaults to {}.
        collision_kwargs (dict, optional): Collision kwargs. Defaults to {}.
    """
    position = position if position is not None else np.zeros(3)
    baseVisualShapeIndex = self.physics_client.createVisualShape(geom_type, **visual_kwargs)
    if not ghost:
        baseCollisionShapeIndex = self.physics_client.createCollisionShape(geom_type, **collision_kwargs)
    else:
        baseCollisionShapeIndex = -1
    self._bodies_idx[body_name] = self.physics_client.createMultiBody(
        baseVisualShapeIndex=baseVisualShapeIndex,
        baseCollisionShapeIndex=baseCollisionShapeIndex,
        baseMass=mass,
        basePosition=position,
    )

    if lateral_friction is not None:
        self.set_lateral_friction(body=body_name, link=-1, lateral_friction=lateral_friction)
    if spinning_friction is not None:
        self.set_spinning_friction(body=body_name, link=-1, spinning_friction=spinning_friction)


def getBaseTransformMatrix(t):
    r""" get base for DH transforms. Invariant to rotation.

    Args:
        t (np.array): xyz coordinate of the base for DH chain transforms

    Returns:
        [np.array]: (1,4,4) translation matrix
    """

    T = np.array([
        [1., 0., 0., t[0]],
        [0., 1., 0., t[1]],
        [0., 0., 1., t[2]],
        [0., 0., 0., 1.]
    ],dtype=np.float64)

    return T[np.newaxis,:]

def getBaseTransformMatrixBatch(t, batch_zeros, batch_ones):
    r""" get base for DH transforms. Invariant to rotation.

    Args:
        t (np.array): xyz coordinate of the base for DH chain transforms

    Returns:
        [np.array]: (B,4,4) translation matrix
    """

    T = np.stack([
        np.stack([batch_ones, batch_zeros, batch_zeros, t[:,0]], axis=1),
        np.stack([batch_zeros, batch_ones, batch_zeros, t[:,1]], axis=1),
        np.stack([batch_zeros, batch_zeros, batch_ones, t[:,2]], axis=1),
        np.stack([batch_zeros, batch_zeros, batch_zeros, batch_ones], axis=1)
    ],axis=1)

    return T
    
def getModifiedTransformMatrix(batch_thetas, a, d, alpha, batch_zeros, batch_ones):
    r"""
    Returns 4x4 homogenous matrix from Modified DH parameters for batch of thetas (i.e., single joint angle).
    """
    cTheta = np.cos(batch_thetas) 
    sTheta = np.sin(batch_thetas)

    # TODO: place these outside
    calpha = np.cos(alpha*batch_ones)
    salpha = np.sin(alpha*batch_ones)


    T = np.stack([
        np.stack([cTheta, -sTheta, batch_zeros, a*batch_ones], axis=1),
        np.stack([ calpha * sTheta, calpha * cTheta, -salpha, -d * salpha], axis=1),
        np.stack([salpha * sTheta, salpha * cTheta, calpha, d * calpha], axis=1),
        np.stack([batch_zeros, batch_zeros, batch_zeros, batch_ones], axis=1)
    ], axis=1)

    return T
