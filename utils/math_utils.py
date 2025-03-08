"""
Quaternion and rotation matrix utilities.
"""

import jax.numpy as jnp
import numpy as np

H = np.concatenate([[np.zeros(3)], np.eye(3)], axis=0)
tmp = -1 * np.eye(4)
tmp[0, 0] = 1
T = tmp

def R(q: jnp.ndarray) -> jnp.ndarray:
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

# def q_to_Q(q):
#     return H.T @ T @ left_q(q) @ T @ left_q(q) @ H

# def rot_2_q(rot: jnp.ndarray) -> jnp.ndarray:
#     factor = 1/jnp.sqrt((1 + jnp.dot(rot,rot)))
#     h1 = jnp.concatenate([[1], rot])

#     return factor * h1


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
    Compute the derivative of the rotation vector to quaternion mapping.
    Args:
        phi (np.ndarray): The rotation vector.
    Returns:
        np.ndarray: The derivative of the rotation vector to quaternion mapping.
    """
    theta = np.linalg.norm(phi) 
    term1 = -0.5 * np.sin(theta/2) * phi.T / theta
    term2 = 0.5 * np.cos(theta/2) * np.outer(phi/theta,phi/theta) + np.sin(theta/2) * (np.eye(3)/theta - np.outer(phi,phi)/theta**3)

    return np.block([[term1], [term2]])

def G(q: np.ndarray) -> np.ndarray:
    """
    Helper function
    """
    return left_q(q) @ H

def rodrigues_rotation_matrix(k, theta):
    """
    Returns the 3x3 rotation matrix that rotates vectors by 'theta' about
    the axis 'k' using Rodrigues' rotation formula.

    Parameters:
    -----------
    k     : array-like, shape (3,)
            The axis of rotation (will be normalized internally).
    theta : float
            The rotation angle in radians.

    Returns:
    --------
    R     : ndarray, shape (3,3)
            The rotation matrix.
    """
    # Ensure k is a numpy array
    k = np.asarray(k, dtype=float)

    # Normalize the axis to get a unit vector
    k = k / np.linalg.norm(k)

    # Create the skew-symmetric matrix [k]_x
    K = np.array([
        [0,      -k[2],   k[1]],
        [k[2],    0,     -k[0]],
        [-k[1],   k[0],   0   ]
    ])

    # Compute the Rodrigues rotation matrix
    I = np.eye(3)
    R = (
        I
        + np.sin(theta) * K
        + (1 - np.cos(theta)) * (K @ K)
    )
    return R
