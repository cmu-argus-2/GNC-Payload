"""
This script estimates the Q and R matrices for the MEKF.
"""

# pylint: disable=import-error
# pylint: disable=unsubscriptable-object
import brahe
import numpy as np
import quaternion
from brahe.epoch import Epoch

from dynamics.ekf_dynamics import EKFDynamics
from dynamics.orbital_att_dynamics import Dynamics
from orbit_determination.ekf import EKF
from orbit_determination.landmark_bearing_sensors import GroundTruthLandmarkBearingSensor
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from sensors.camera_model import CameraModelManager
from sensors.imu import IMU
from utils.config_utils import load_config
from utils.orbit_utils import get_sso_orbit_state

# 1. Tuning of Q matrices: Var from delta between estimated and true trajectory
CONFIG = load_config()
# Set the world update rate and mission duration to a rate that is workable for testing
CONFIG["solver"]["world_update_rate"] = 2  # Hz
CONFIG["mission"]["duration"] = 90 * 60  # s

DT = 1 / CONFIG["solver"]["world_update_rate"]
STARTING_EPOCH = Epoch(*brahe.time.mjd_to_caldate(CONFIG["mission"]["start_date"]))
N = int(np.ceil(CONFIG["mission"]["duration"] / DT))  # number of time steps in the simulation

LANDMARK_BEARING_SENSOR = GroundTruthLandmarkBearingSensor()
CAMERA_MODEL_MANAGER = CameraModelManager()
DATA_MANAGER = ODSimulationDataManager(STARTING_EPOCH, DT)

INITIAL_STATE = get_sso_orbit_state(STARTING_EPOCH, 0, -73, 510e3, northwards=True)
INITIAL_STATE = INITIAL_STATE / 1e3  # Convert from m to km and m/s to km/s
# Set the initial rotation matrix to identity
INIT_ROT = np.eye(3)

# Fix a constant rotation velocity for the test.
INIT_OMEGA = np.array([0, 0, np.pi / 18])
# [TODO:] possibly disperse
# INIT_OMEGA = INIT_OMEGA

DATA_MANAGER.push_next_state(INITIAL_STATE, INIT_ROT, INIT_OMEGA)

# Apply error to INIT_ROT and ensure orthonormality
NOISY_ROT = INIT_ROT

# Set the number of update iterations for the IEKF
NUM_ITER = 1

# Set up scaling parameter for the unmodelled acceleration
UA_SCALE = 1

# Set up scaling parameter for gyro bias
GYRO_BIAS_SCALE = 1


""""""
# Prep Q matrix for the EKF.
Q = np.eye(16) * 1e-16
# Unmodelled acceleration has larger uncertainty
Q[6:9, 6:9] = np.eye(3) * 1e-12
# # Bias uncertainty also larger
Q[13:16, 13:16] = np.eye(3) * 1e-12

P = np.diag(
    [5e-3] * 3  # r
    + [5e-3] * 3  # v
    + [1e-4] * 3  # ua
    + [1e-4]  # drag
    + [1e-4] * 3  # quaternion
    + [1e-4] * 3  # gyro bias
)


# 1.1. EKF dynamics model for orbit determination
EKF_DYNAMICS = EKFDynamics(
    config=CONFIG,
    use_drag=False,
    use_j2=True,
    use_j34=False,
    use_sun_grav=False,
    use_moon_grav=False,
    use_unmodelled_a=True,
    use_drag_scalar=True,
    ua_scale=UA_SCALE,
)

# 1.2. True model for orbit determination
GROUND_TRUTH_DYNAMICS = Dynamics(
    config=CONFIG,
    use_drag=True,
    use_j2=True,
    use_j34=False,
    use_sun_grav=True,
    use_moon_grav=True,
)

# 1.3. define a true state trajectory
IMU_SENSOR = IMU.get_default_imu(DT)
IMU_GYRO_BIAS = IMU_SENSOR.get_bias()[0] * GYRO_BIAS_SCALE
EKF_FILTER = EKF(
    # error ranges are in meters and m/s
    r=INITIAL_STATE[0:3],
    v=INITIAL_STATE[3:6],
    ua=np.random.normal(0, 1e-8, 3) * UA_SCALE,
    q=quaternion.as_float_array(quaternion.from_rotation_matrix(NOISY_ROT)),
    P=P,
    Q=Q,
    dt=DT,
    config=CONFIG,
    ekf_dynamics=EKF_DYNAMICS,
    w_b=IMU_GYRO_BIAS,
    gyro_bias_scale=GYRO_BIAS_SCALE,
)
EKF_FILTER.drag_est = GROUND_TRUTH_DYNAMICS.drag_constant(
    x=INITIAL_STATE[0:6], epoch=STARTING_EPOCH
)
EKF_FILTER.ua = GROUND_TRUTH_DYNAMICS.unmodelled_acceleration(x=INITIAL_STATE, epoch=STARTING_EPOCH)

# Store values
TRUE_X = []
TRUE_EKF_X = []
EST_X = []
CUR_EPOCH = STARTING_EPOCH
for t in range(0, N - 1):
    # take a set of measurements every minute
    X = DATA_MANAGER.latest_state
    X = np.concatenate([X, EKF_FILTER.ua])
    QUAT = quaternion.from_rotation_matrix(DATA_MANAGER.latest_attitude)
    W = DATA_MANAGER.latest_angular_velocity
    X_FULL = np.array(
        np.concatenate([X[:6], QUAT.components, W])
    )  # pylint: disable=E1136  # pylint/issues/9590

    # set previous ekf states to true states
    EKF_FILTER.r_m = X[0:3]  # pylint: disable=E1136  # pylint/issues/9590
    EKF_FILTER.v_m = X[3:6]  # pylint: disable=E1136  # pylint/issues/9590
    EKF_FILTER.ua = GROUND_TRUTH_DYNAMICS.unmodelled_acceleration(x=X_FULL, epoch=CUR_EPOCH)
    EKF_FILTER.drag_est = GROUND_TRUTH_DYNAMICS.drag_constant(
        x=X[0:6], epoch=CUR_EPOCH
    )  # pylint: disable=E1136  # pylint/issues/9590
    EKF_FILTER.q_m = X_FULL[6:10]  # pylint: disable=E1136  # pylint/issues/9590
    EKF_FILTER.w_b = IMU_GYRO_BIAS

    EKF_FILTER.predict(u=W, epoch=CUR_EPOCH)

    NEXT_EST_X = np.concatenate(
        [
            EKF_FILTER.r_p,
            EKF_FILTER.v_p,
            EKF_FILTER.ua,
            [EKF_FILTER.drag_est],
            EKF_FILTER.q_p,
            EKF_FILTER.w_b,
        ]
    )

    # propagate true states
    # Get a gyro measurement to use in the EKF and the current gyro bias for the ground truth
    CUR_EPOCH = DATA_MANAGER.latest_epoch
    GYRO_MEAS, _ = IMU_SENSOR.update(W, np.zeros((3)))
    IMU_GYRO_BIAS = IMU_SENSOR.get_bias()[0]
    NEXT_STATE = GROUND_TRUTH_DYNAMICS.perturbed_f(x=X_FULL, dt=DT, epoch=CUR_EPOCH)
    NEXT_QUAT = quaternion.quaternion(*NEXT_STATE[6:10])
    NEXT_OMEGA = NEXT_STATE[10:13]

    # next true ekf states
    NEXT_TRUE_EKF_X = np.concatenate(
        [
            NEXT_STATE[0:6],
            GROUND_TRUTH_DYNAMICS.unmodelled_acceleration(x=X, epoch=CUR_EPOCH),
            [
                GROUND_TRUTH_DYNAMICS.drag_constant(x=X[0:6], epoch=CUR_EPOCH)
            ],  # pylint: disable=E1136  # pylint/issues/9590
            NEXT_STATE[6:10],
            IMU_GYRO_BIAS,
        ]
    )

    TRUE_X.append(NEXT_STATE)
    TRUE_EKF_X.append(NEXT_TRUE_EKF_X)
    EST_X.append(NEXT_EST_X)

    DATA_MANAGER.push_next_state(
        NEXT_STATE[0:6], quaternion.as_rotation_matrix(NEXT_QUAT), NEXT_OMEGA
    )
    if t % 100 == 0:
        print(f"Step {t+1}/{N-1} completed")

# Save arrays to a single file for later analysis
np.savez(
    "trajectory_data.npz",
    true_x=np.array(TRUE_X),
    true_ekf_x=np.array(TRUE_EKF_X),
    est_x=np.array(EST_X),
)

# 1.4. compute estimated xk+1 along trajectory

# 1.5. variance between estimated and true trajectory

# 2. Tuning of R matrices: Var from delta between estimated and true measurement vectors

# 2.1. EKF measurement model for orbit determination
