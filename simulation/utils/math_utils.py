"""
Quaternion and rotation matrix utilities.
"""

# pylint: disable=import-error
import jax.numpy as jnp
import numpy as np


def quat2rotm(q: jnp.ndarray) -> jnp.ndarray:
    """Return the rotation matrix corresponding to the quaternion q.

    Args:
        q (np.array): The quaternion to convert.

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
        rot (np.ndarray): The rotation vector to convert.

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


def der_rp2q(phi: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the rotation vector to quaternion mapping.
    Args:
        phi (np.ndarray): The rotation vector.
    Returns:
        np.ndarray: The derivative of the rotation vector to quaternion mapping.
    """
    theta = np.linalg.norm(phi)
    term1 = -0.5 * np.sin(theta / 2) * phi.T / theta
    term2 = 0.5 * np.cos(theta / 2) * np.outer(phi / theta, phi / theta) + np.sin(theta / 2) * (
        np.eye(3) / theta - np.outer(phi, phi) / theta**3
    )

    return np.block([[term1], [term2]])


def left_q_3(q: np.ndarray) -> np.ndarray:
    """
    Helper function
    """
    H = np.concatenate([[np.zeros(3)], np.eye(3)], axis=0)
    return left_q(q) @ H


def skew(r: np.ndarray) -> np.ndarray:
    """
    Compute the skew-symmetric matrix of a vector.

    Args:
        r (np.ndarray): The vector to convert to a skew-symmetric matrix.

    Returns:
        np.ndarray: The skew-symmetric matrix.
    """
    return np.array(
        [
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0],
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Compute the conjugate of a quaternion.

    Args:
        q (np.ndarray): The quaternion [w, x, y, z].

    Returns:
        np.ndarray: The conjugate quaternion [w, -x, -y, -z].
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_2_rot(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to a rotation vector.

    Args:
        q (np.ndarray): The quaternion to convert.

    Returns:
        jnp.ndarray: The corresponding rotation matrix as a jax array.
    """
    # Normalize the quaternion
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q

    # Compute the rotation vector
    theta = 2.0 * jnp.arccos(q[0])
    axis = jnp.array([q[1], q[2], q[3]])
    axis = (
        axis / jnp.linalg.norm(axis) if jnp.linalg.norm(axis) > 1e-8 else jnp.array([1.0, 0.0, 0.0])
    )

    return theta * axis
