"""
This script estimates the Q and R matrices for the MEKF.
State noise compensation.
"""

# pylint: disable=import-error
# pylint: disable=unsubscriptable-object
import os

import brahe
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from brahe.epoch import Epoch
from dynamics.ekf_dynamics import EKFDynamics
from dynamics.orbital_att_dynamics import Dynamics
from sensors.camera_model import CameraModelManager
from sensors.imu import IMU
from utils.config_utils import load_config
from utils.orbit_utils import get_sso_orbit_state

from orbit_determination.ekf import EKF
from orbit_determination.landmark_bearing_sensors import GroundTruthLandmarkBearingSensor
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager

TRAJ_DATA_FOLDER = "results/mekf_tuning"
TRAJ_DATA_FILE = os.path.join(TRAJ_DATA_FOLDER, "trajectory_data.npz")

if not os.path.exists(TRAJ_DATA_FILE):
    print(f"{TRAJ_DATA_FILE} not found. Running simulation to generate data...")

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

    # Set up scaling parameter
    variable_scaling = np.array(
        [1e-3] * 3  # r
        + [1e2] * 3  # v
        + [1e8] * 3  # ua
        + [1]  # drag
        + [1e1] * 3  # quaternion
        + [1e2] * 3  # gyro bias
    )
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
        state_scaling=variable_scaling,
    )
    EKF_FILTER.drag_est = GROUND_TRUTH_DYNAMICS.drag_constant(
        x=INITIAL_STATE[0:6], epoch=STARTING_EPOCH
    )
    EKF_FILTER.ua = GROUND_TRUTH_DYNAMICS.unmodelled_acceleration(
        x=INITIAL_STATE, epoch=STARTING_EPOCH
    )

    # Store values
    TRUE_X_LIST = []
    TRUE_EKF_X_LIST = []
    EST_X_LIST = []
    CUR_EPOCH = STARTING_EPOCH
    TIMES_LIST = []
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
                EKF_FILTER.drag_est,
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
                GROUND_TRUTH_DYNAMICS.drag_constant(
                    x=X[0:6], epoch=CUR_EPOCH
                ),  # pylint: disable=E1136  # pylint/issues/9590
                NEXT_STATE[6:10],
                IMU_GYRO_BIAS,
            ]
        )

        TRUE_X_LIST.append(NEXT_STATE)
        TRUE_EKF_X_LIST.append(NEXT_TRUE_EKF_X)
        EST_X_LIST.append(NEXT_EST_X)
        TIMES_LIST.append(t * DT)

        DATA_MANAGER.push_next_state(
            NEXT_STATE[0:6], quaternion.as_rotation_matrix(NEXT_QUAT), NEXT_OMEGA
        )
        if t % 100 == 0:
            print(f"Step {t+1}/{N-1} completed")

    TRUE_X = np.array(TRUE_X_LIST)
    TRUE_EKF_X = np.array(TRUE_EKF_X_LIST)
    EST_X = np.array(EST_X_LIST)
    TIMES = np.array(TIMES_LIST)
    # Save arrays to a single file for later analysis
    os.makedirs(TRAJ_DATA_FOLDER, exist_ok=True)
    np.savez(
        TRAJ_DATA_FILE,
        true_x=TRUE_X,
        true_ekf_x=TRUE_EKF_X,
        est_x=EST_X,
        times=TIMES,
    )

# Load trajectory data
DATA = np.load(TRAJ_DATA_FILE)
TRUE_X = DATA["true_x"]
TRUE_EKF_X = DATA["true_ekf_x"]
EST_X = DATA["est_x"]
TIMES = DATA["times"]

# Plotting the true state trajectory
# true states
# [0:3] for position, [3:6] for velocity
plt.figure(figsize=(10, 6))
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(TRUE_X[:, 0], TRUE_X[:, 1], TRUE_X[:, 2], label="True Trajectory", color="blue")
ax.set_xlabel("X Position (km)")
ax.set_ylabel("Y Position (km)")
ax.set_zlabel("Z Position (km)")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(TRAJ_DATA_FOLDER, "true_trajectory_3d.png"))

fig2, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
labels = ["X", "Y", "Z"]
for i in range(3):
    axs[i].plot(TIMES, TRUE_X[:, i], label=f"True {labels[i]}")
    axs[i].set_ylabel(f"{labels[i]} Position (km)")
    axs[i].legend()
axs[2].set_xlabel("Time (s)")
plt.tight_layout()
fig2.savefig(os.path.join(TRAJ_DATA_FOLDER, "true_trajectory_xyz.png"))

fig3, axs_v = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
vel_labels = ["X", "Y", "Z"]
for i in range(3):
    axs_v[i].plot(TIMES, TRUE_X[:, 3 + i], label=f"True {vel_labels[i]} Velocity")
    axs_v[i].set_ylabel(f"{vel_labels[i]} Velocity (km/s)")
    axs_v[i].legend()
axs_v[2].set_xlabel("Time (s)")
plt.tight_layout()
fig3.savefig(os.path.join(TRAJ_DATA_FOLDER, "true_velocity_xyz.png"))

fig4, axs_q = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
quat_labels = ["q0", "q1", "q2", "q3"]
for i in range(4):
    axs_q[i].plot(TIMES, TRUE_X[:, 6 + i], label=f"True {quat_labels[i]}")
    axs_q[i].set_ylabel(f"{quat_labels[i]}")
    axs_q[i].legend()
axs_q[3].set_xlabel("Time (s)")
plt.tight_layout()
fig4.savefig(os.path.join(TRAJ_DATA_FOLDER, "true_quaternion.png"))

fig5, axs_w = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
ang_vel_labels = ["X", "Y", "Z"]
for i in range(3):
    axs_w[i].plot(TIMES, TRUE_X[:, 10 + i], label=f"True {ang_vel_labels[i]} Angular Velocity")
    axs_w[i].set_ylabel(f"{ang_vel_labels[i]} (rad/s)")
    axs_w[i].legend()
axs_w[2].set_xlabel("Time (s)")
plt.tight_layout()
fig5.savefig(os.path.join(TRAJ_DATA_FOLDER, "true_angular_velocity_xyz.png"))

# ekf states
fig_ua, axs_ua = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
ua_labels = ["X", "Y", "Z"]
for i in range(3):
    axs_ua[i].plot(TIMES, TRUE_EKF_X[:, 6 + i], label=f"True UA {ua_labels[i]}")
    axs_ua[i].set_ylabel(f"UA {ua_labels[i]} (km/s²)")
    axs_ua[i].legend()
axs_ua[2].set_xlabel("Time (s)")
plt.tight_layout()
fig_ua.savefig(os.path.join(TRAJ_DATA_FOLDER, "true_unmodelled_acceleration.png"))

fig_drag, ax_drag = plt.subplots(figsize=(12, 4))
ax_drag.plot(TIMES, TRUE_EKF_X[:, 9], label="True Drag Factor")
ax_drag.set_ylabel("Drag Factor")
ax_drag.set_xlabel("Time (s)")
ax_drag.legend()
plt.tight_layout()
fig_drag.savefig(os.path.join(TRAJ_DATA_FOLDER, "true_drag_factor.png"))

fig_bias, axs_bias = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
bias_labels = ["X", "Y", "Z"]
for i in range(3):
    axs_bias[i].plot(TIMES, TRUE_EKF_X[:, 14 + i], label=f"True Gyro Bias {bias_labels[i]}")
    axs_bias[i].set_ylabel(f"Gyro Bias {bias_labels[i]} (rad/s)")
    axs_bias[i].legend()
axs_bias[2].set_xlabel("Time (s)")
plt.tight_layout()
fig_bias.savefig(os.path.join(TRAJ_DATA_FOLDER, "true_gyro_bias.png"))

# delta between estimated and true trajectory
# position error
# Compute errors
POS_ERROR = np.linalg.norm(EST_X[:, 0:3] - TRUE_EKF_X[:, 0:3], axis=1)
VEL_ERROR = np.linalg.norm(EST_X[:, 3:6] - TRUE_EKF_X[:, 3:6], axis=1)
UA_ERROR = np.linalg.norm(EST_X[:, 6:9] - TRUE_EKF_X[:, 6:9], axis=1)
DRAG_ERROR = np.abs(EST_X[:, 9] - TRUE_EKF_X[:, 9])
QUAT_ERROR = np.linalg.norm(EST_X[:, 10:14] - TRUE_EKF_X[:, 10:14], axis=1)
GYRO_BIAS_ERROR = np.linalg.norm(EST_X[:, 14:] - TRUE_EKF_X[:, 14:], axis=1)

FIG_ERR, AXs_ERR = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
AXs_ERR[0].plot(TIMES, POS_ERROR, label="Position Error (km)")
AXs_ERR[0].set_ylabel("Position Error (km)")
AXs_ERR[0].legend()

# velocity error
AXs_ERR[1].plot(TIMES, VEL_ERROR, label="Velocity Error (km/s)")
AXs_ERR[1].set_ylabel("Velocity Error (km/s)")
AXs_ERR[1].legend()

# unmodelled acceleration
AXs_ERR[2].plot(TIMES, UA_ERROR, label="Unmodelled Accel. Error (km/s²)")
AXs_ERR[2].set_ylabel("UA Error (km/s²)")
AXs_ERR[2].legend()

# drag factor error
AXs_ERR[3].plot(TIMES, DRAG_ERROR, label="Drag Factor Error")
AXs_ERR[3].set_ylabel("Drag Error")
AXs_ERR[3].legend()

# quaternion error
AXs_ERR[4].plot(TIMES, QUAT_ERROR, label="Quaternion Error")
AXs_ERR[4].set_ylabel("Quat Error")
AXs_ERR[4].legend()

# gyro bias error
AXs_ERR[5].plot(TIMES, GYRO_BIAS_ERROR, label="Gyro Bias Error (rad/s)")
AXs_ERR[5].set_ylabel("Gyro Bias Error (rad/s)")
AXs_ERR[5].set_xlabel("Time (s)")
AXs_ERR[5].legend()
plt.tight_layout()
FIG_ERR.savefig(os.path.join(TRAJ_DATA_FOLDER, "state_errors.png"))

plt.show()

# Compute variances
POS_VAR = np.var(EST_X[:, 0:3] - TRUE_EKF_X[:, 0:3], axis=0)
VEL_VAR = np.var(EST_X[:, 3:6] - TRUE_EKF_X[:, 3:6], axis=0)
UA_VAR = np.var(EST_X[:, 6:9] - TRUE_EKF_X[:, 6:9], axis=0)
DRAG_VAR = np.var(EST_X[:, 9] - TRUE_EKF_X[:, 9])
QUAT_VAR = np.var(EST_X[:, 10:14] - TRUE_EKF_X[:, 10:14], axis=0)
GYRO_BIAS_VAR = np.var(EST_X[:, 14:] - TRUE_EKF_X[:, 14:], axis=0)
print("Estimated State Variances:")
print(f"Position Variance (km²): {POS_VAR}")
print(f"Velocity Variance (km²/s²): {VEL_VAR}")
print(f"Unmodelled Acceleration Variance (km²/s^4): {UA_VAR}")
print(f"Drag Factor Variance: {DRAG_VAR}")
print(f"Quaternion Variance: {QUAT_VAR}")
print(f"Gyro Bias Variance (rad²/s²): {GYRO_BIAS_VAR}")
# Construct the Q matrix
Q_EST = np.zeros((17, 17))
# Q_EST[0:3, 0:3] = np.diag(POS_VAR)
# Q_EST[3:6, 3:6] = np.diag(VEL_VAR)
Q_EST[6:9, 6:9] = np.diag(UA_VAR)
Q_EST[9, 9] = DRAG_VAR
# Q_EST[10:14, 10:14] = np.diag(QUAT_VAR)
Q_EST[14:17, 14:17] = np.diag(GYRO_BIAS_VAR)
print("Estimated Q Matrix:")
print(Q_EST)
# Save the Q matrix to a file
np.savez(os.path.join(TRAJ_DATA_FOLDER, "estimated_Q_matrix.npz"), Q=Q_EST)
# period of the unmodelled accelerations is half the orbit period
# ideally the unmodelled accelerations would be a 2nd order gauss markov process
# in the current setting it's considered a dynamic model compensation term
# with a 1st order. Because the mean unmodelled acceleration is close to zero
# it's worth setting up a decay term to bring the unmodelled acceleration estimate
# back to zero over time.
# It is set at ~1/8 of the orbit period, tau = 700s
