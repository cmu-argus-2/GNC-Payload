"""
Quaternion and rotation matrix utilities.
"""

import jax.numpy as jnp
import numpy as np


def R(q: jnp.ndarray) -> jnp.ndarray:
    """Return the rotation matrix corresponding to the quaternion q.

    Args:
        q (jnp.array): The quaternion to convert.

    Returns:
        jnp.ndarray: The corresponding rotation matrix as jax array.
    """
    # Convert quaternion to rotation matrix
    R = jnp.array(
        [
            [
                1 - 2 * (q[2] ** 2 + q[3] ** 2),
                2 * (q[1] * q[2] - q[0] * q[3]),
                2 * (q[1] * q[3] + q[0] * q[2]),
            ],
            [
                2 * (q[1] * q[2] + q[0] * q[3]),
                1 - 2 * (q[1] ** 2 + q[3] ** 2),
                2 * (q[2] * q[3] - q[0] * q[1]),
            ],
            [
                2 * (q[1] * q[3] - q[0] * q[2]),
                2 * (q[2] * q[3] + q[0] * q[1]),
                1 - 2 * (q[1] ** 2 + q[2] ** 2),
            ],
        ]
    )
    return R


def rot_2_q(rot: jnp.ndarray) -> jnp.ndarray:
    """
    Convert a rotation vector to a quaternion.

    Args:
        rot (jnp.ndarray): The rotation vector to convert.

    Returns:
        jnp.ndarray: The corresponding quaternion as a jax array.
    """

    # Normalize the rotation vector
    theta = jnp.linalg.norm(rot)
    if theta < 1e-8:
        return jnp.array([1.0, 0.0, 0.0, 0.0])

    # Compute the quaternion components
    half_theta = theta / 2.0
    q = jnp.array(
        [
            jnp.cos(half_theta),
            rot[0] * jnp.sin(half_theta) / theta,
            rot[1] * jnp.sin(half_theta) / theta,
            rot[2] * jnp.sin(half_theta) / theta,
        ]
    )
    return q


def left_q(q: np.ndarray) -> np.ndarray:
    """
    Left multiplication of quaternion q.

    Args:
        q (np.ndarray): The quaternion to turn into a left multiply.

    Returns:
        np.ndarray: The left multiply matrix of the quaternion.

    """
    return np.array(
        [
            [q[0], -q[1], -q[2], -q[3]],
            [q[1], q[0], -q[3], q[2]],
            [q[2], q[3], q[0], -q[1]],
            [q[3], -q[2], q[1], q[0]],
        ]
    )


def right_q(q: np.ndarray) -> np.ndarray:
    """
    Right multiplication of quaternion q.

    Args:
        q (np.ndarray): The quaternion to turn into a right multiply.

    Returns:
        np.ndarray: The right multiply matrix of the quaternion.
    """
    return np.array(
        [
            [q[0], -q[1], -q[2], -q[3]],
            [q[1], q[0], q[3], -q[2]],
            [q[2], -q[3], q[0], q[1]],
            [q[3], q[2], -q[1], q[0]],
        ]
    )

def Drp2q(phi: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of a 3D rotation vector to quaternion mapping.
    Args:
        phi (np.ndarray): The rotation vector.
    Returns:
        np.ndarray: The derivative.
    """
    frac1 = 1/(np.sqrt(1 + np.dot(phi, phi)))
    H1 = np.concatenate([[np.zeros(3)], np.eye(3)], axis=0)
    frac2 = (1 + np.dot(phi, phi))**(-3/2)
    H2 = np.outer(np.append(1,phi), phi)

    return frac1*H1 - frac2*H2

def G(q: np.ndarray) -> np.ndarray:
    """
    Helper function    
    """
    H = np.concatenate([[np.zeros(3)], np.eye(3)], axis=0)
    return left_q(q) @ H
