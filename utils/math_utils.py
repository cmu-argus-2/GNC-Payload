import jax.numpy as jnp


def R(q):
    """Return the rotation matrix corresponding to the quaternion q.

    Args:
        q (jnp.array): The quaternion to convert.

    Returns:
        np.ndarray: The corresponding rotation matrix.
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


def rot_2_q(rot):
    """
    Convert a rotation vector to a quaternion.
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
