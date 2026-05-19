"""Lightweight orbital + attitude dynamics compatibility module.

This shim supports legacy imports from orbit_determination scripts that expect
`simulation.dynamics.orbital_att_dynamics`.
"""

import numpy as np
import quaternion
from brahe.constants import GM_EARTH

from dynamics.drag_dynamics import drag_dynamics
from dynamics.grav_potential_dynamics import j2_dynamics, j3_dynamics, j4_dynamics
from dynamics.orbital_dynamics import Dynamics as OrbitalDynamics
from dynamics.third_body_dynamics import moon_gravity, sun_gravity


class DynamicsIDX:  # pylint: disable=too-few-public-methods
    """State index helper for [orb(6), quat(4), omega(3), gyro_bias(3 optional)]."""

    # Class-level defaults used by legacy code paths that access dynidx.QUAT directly.
    ORB = slice(0, 6)
    QUAT = slice(6, 10)
    OMEGA = slice(10, 13)
    GYR_BIAS = slice(13, 16)
    NX = 16

    RX = 0
    RY = 1
    RZ = 2
    VX = 3
    VY = 4
    VZ = 5
    QW = 6
    QX = 7
    QY = 8
    QZ = 9
    OMEGA_X = 10
    OMEGA_Y = 11
    OMEGA_Z = 12

    def __init__(self, has_gyro_bias: bool = False):
        self.has_gyro_bias = has_gyro_bias
        self.ORB = slice(0, 6)
        self.QUAT = slice(6, 10)
        self.OMEGA = slice(10, 13)
        if has_gyro_bias:
            self.GYR_BIAS = slice(13, 16)
            self.NX = 16
        else:
            self.GYR_BIAS = slice(13, 13)
            self.NX = 13


class Dynamics:
    """Orbital + simple attitude/gyro-bias propagation compatibility class."""

    def __init__(
        self,
        config: dict,
        use_drag: bool,
        use_j2: bool,
        use_j34: bool,
        use_sun_grav: bool,
        use_moon_grav: bool,
        include_gyro_bias: bool = False,
        gyro_bias_tau: float = 1.0,
        gyro_bias_std: float = 0.0,
    ) -> None:
        self.idx = DynamicsIDX(has_gyro_bias=include_gyro_bias)
        self.include_gyro_bias = include_gyro_bias
        self.gyro_bias_tau = float(max(gyro_bias_tau, 1e-6))
        self.gyro_bias_std = float(max(gyro_bias_std, 0.0))

        self.use_drag = use_drag
        self.use_j2 = use_j2
        self.use_j34 = use_j34
        self.use_sun_grav = use_sun_grav
        self.use_moon_grav = use_moon_grav
        self.orbital = OrbitalDynamics(
            config=config,
            use_drag=use_drag,
            use_j2=use_j2,
            use_j34=use_j34,
            use_sun_grav=use_sun_grav,
            use_moon_grav=use_moon_grav,
        )

    def perturbed_f(self, x: np.ndarray, dt: float, epoch=None) -> np.ndarray:
        """Propagate orbital state, quaternion, angular velocity, and optional gyro bias."""
        x = np.asarray(x)
        orb = x[self.idx.ORB]
        quat_arr = x[self.idx.QUAT]
        omega = x[self.idx.OMEGA]

        # OrbitalDynamics expects SI units (m, m/s), while OD state here is km, km/s.
        orb_m = np.concatenate([orb[:3] * 1e3, orb[3:6] * 1e3])
        orb_next_m = self.orbital.perturbed_f(orb_m, dt, epoch)
        orb_next = np.concatenate([orb_next_m[:3] / 1e3, orb_next_m[3:6] / 1e3])

        # Guard against unit regressions: LEO radius should remain on order of 10^3-10^4 km.
        r_km = np.linalg.norm(orb_next[:3])
        if not (6.0e3 <= r_km <= 5.0e4):
            raise ValueError(
                f"Unphysical orbital radius after propagation: {r_km:.3f} km. "
                "Check km<->m conversions in dynamics propagation."
            )

        q = quaternion.from_float_array(quat_arr)
        dq = quaternion.from_rotation_vector(omega * dt)
        q_next_arr = quaternion.as_float_array(q * dq)
        q_next_arr = q_next_arr / np.linalg.norm(q_next_arr)

        if self.include_gyro_bias:
            bias = x[self.idx.GYR_BIAS]
            phi = np.exp(-dt / self.gyro_bias_tau)
            sigma = self.gyro_bias_std * np.sqrt(max(0.0, 1.0 - phi * phi))
            bias_next = phi * bias + np.random.normal(0.0, sigma, 3)
            return np.concatenate([orb_next, q_next_arr, omega, bias_next])

        return np.concatenate([orb_next, q_next_arr, omega])

    def get_accel_components(self, x: np.ndarray, epoch=None) -> dict[str, np.ndarray]:
        """Return per-effect acceleration components in km/s^2 for diagnostics/plots."""
        r = np.asarray(x[:3])
        v = np.asarray(x[3:6])
        r_norm = np.linalg.norm(r)

        earth = -r * GM_EARTH / (r_norm ** 3)

        j2 = j2_dynamics(r) if self.use_j2 else np.zeros(3)
        j34 = (j3_dynamics(r) + j4_dynamics(r)) if self.use_j34 else np.zeros(3)
        drag = drag_dynamics(x=np.concatenate([r, v]), drag_const=self.orbital.drag_const, latest_epoch=epoch) if (self.use_drag and epoch is not None) else np.zeros(3)
        sun = sun_gravity(r_sat=r, epoch=epoch) if (self.use_sun_grav and epoch is not None) else np.zeros(3)
        moon = moon_gravity(r_sat=r, epoch=epoch) if (self.use_moon_grav and epoch is not None) else np.zeros(3)

        return {
            "earth_gravity": earth,
            "j2": j2,
            "j34": j34,
            "drag": drag,
            "sun_gravity": sun,
            "moon_gravity": moon,
        }
