"""
This module provides post-processing and visualization tools for analyzing the
results of an Extended Kalman Filter (EKF) applied to orbit determination
problems. It loads error metrics and filter outputs from NumPy files in a
specified output directory, and generates a series of plots to assess EKF
performance. These include position and velocity errors, confidence intervals,
unmodeled acceleration, gyro bias errors, covariance trace, actual gyro bias,
and drag estimates. The generated plots are saved as PNG files in the output
directory for further inspection.

Files loaded:
    - pos_error.npy: Position estimation errors (x, y, z)
    - vel_error.npy: Velocity estimation errors (x, y, z)
    - sigma_high.npy: Upper confidence bounds for position errors
    - sigma_low.npy: Lower confidence bounds for position errors
    - ua_error.npy: Unmodeled acceleration errors
    - gyro_bias_error.npy: Gyro bias estimation errors
    - actual_bias.npy: Actual gyro bias values
    - drag_estimate.npy: Estimated drag values
    - cov_trace.npy: Trace of the EKF covariance matrix

Plots generated:
    1. Position Error with Confidence Intervals
    2. EKF Position Error
    3. EKF Position Error with Confidence Intervals
    4. EKF Velocity Error
    5. EKF Unmodeled Acceleration Error
    6. EKF Gyro Bias Error
    7. EKF Covariance Trace
    8. Actual Gyro Bias
    9. EKF Drag Estimate

All plots are saved in the corresponding output directory.
"""

# pylint: disable=E1136  # pylint/issues/9590
# remove later
# mypy: disable-error-code=index
# mypy: disable-error-code=name-defined
# mypy: disable-error-code=misc
# pylint: disable=import-error,invalid-name,undefined-variable
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.math_utils import left_q

INDEX_TRIAL = 0
DIR_NAME = f"results/ekf_realdrag/trial_{INDEX_TRIAL}"
EKF_STATE = np.load(os.path.join(DIR_NAME, "ekf_state.npy"))
EKF_STATE_STD = np.load(os.path.join(DIR_NAME, "ekf_state_std.npy"))
TRUE_STATE = np.load(os.path.join(DIR_NAME, "true_state.npy"))
COV_TRACE = np.load(os.path.join(DIR_NAME, "cov_trace.npy"))
COV_COND_NUM = np.load(os.path.join(DIR_NAME, "cov_cond_num.npy"))

IDX_DICT = {
    "Position": {"y_lbls": ["x", "y", "z"], "y_units": "km", "idx": slice(0, 3)},
    "Velocity": {"y_lbls": ["vx", "vy", "vz"], "y_units": "km/s", "idx": slice(3, 6)},
    "Unmodelled Acceleration": {
        "y_lbls": ["ax", "ay", "az"],
        "y_units": "km/s^2",
        "idx": slice(6, 9),
    },
    "Drag Scalar": {"y_lbls": ["drag"], "y_units": "-", "idx": 9},
    "Quaternion": {"y_lbls": ["q0", "q1", "q2", "q3"], "y_units": "", "idx": slice(10, 14)},
    "Angular Velocity": {"y_lbls": ["wx", "wy", "wz"], "y_units": "rad/s", "idx": slice(14, 17)},
    "Gyro Bias": {"y_lbls": ["bx", "by", "bz"], "y_units": "rad/s", "idx": slice(17, 20)},
}

ERROR = TRUE_STATE - EKF_STATE
# [TODO]: quaternion error
ERROR[:, 10:14] = left_q(TRUE_STATE[:, 10:14]) @ EKF_STATE[:, 10:14]
# Plot errors for each state variable in IDX_DICT
for key, val in IDX_DICT.items():
    idx = val["idx"]
    y_lbls = val["y_lbls"]
    y_units = val["y_units"]
    # Handle slice (multiple variables) or int (single variable)
    if isinstance(idx, slice):
        num_entries = idx.stop - idx.start
        # State plots
        fig, axes = plt.subplots(num_entries, 1, figsize=(8, 3 * num_entries), sharex=True)
        if num_entries == 1:
            axes = [axes]
        for i in range(num_entries):
            axes[i].plot(TRUE_STATE[:, idx.start + i], label="True")
            axes[i].plot(EKF_STATE[:, idx.start + i], label="EKF")
            # axes[i].set_ylabel(f"{y_lbls[i]} [{y_units}]")
            # axes[i].set_title(f"{key} {y_lbls[i]}")
            if i == num_entries - 1:
                axes[i].legend()
        axes[-1].set_xlabel("Time step")
        fig.tight_layout()
        fig.savefig(os.path.join(DIR_NAME, f"ekf_{key.lower().replace(' ', '_')}_error.png"))
        plt.close(fig)
        # EKF error plots
        fig, axes = plt.subplots(num_entries, 1, figsize=(8, 3 * num_entries), sharex=True)
        if num_entries == 1:
            axes = [axes]
        for i in range(num_entries):
            axes[i].plot(ERROR[:, idx.start + i])
            # axes[i].set_ylabel(f"{y_lbls[i]} [{y_units}]")
            # axes[i].set_title(f"EKF {key} {y_lbls[i]} Error")
        axes[-1].set_xlabel("Time step")
        fig.tight_layout()
        fig.savefig(os.path.join(DIR_NAME, f"ekf_{key.lower().replace(' ', '_')}_error.png"))
        plt.close(fig)
    else:
        fig, ax = plt.subplots()
        ax.plot(ERROR[:, idx])
        ax.set_xlabel("Time step")
        # ax.set_ylabel(f"{y_lbls[0]} error [{y_units}]")
        # ax.set_title(f"EKF {key} {y_lbls[0]} Error")
        fig.savefig(os.path.join(DIR_NAME, f"ekf_{key.lower().replace(' ', '_')}_error.png"))
        plt.close(fig)


# 1. Position Error with Confidence
fig1, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(ERROR[:, 0], label="x")
ax[1].plot(ERROR[:, 1], label="y")
ax[2].plot(ERROR[:, 2], label="z")
ax[0].plot(3 * EKF_STATE_STD[:, 0], "r--")
ax[0].plot(-3 * EKF_STATE_STD[:, 0], "r--")
ax[1].plot(3 * EKF_STATE_STD[:, 1], "r--")
ax[1].plot(-3 * EKF_STATE_STD[:, 1], "r--")
ax[2].plot(3 * EKF_STATE_STD[:, 2], "r--")
ax[2].plot(-3 * EKF_STATE_STD[:, 2], "r--")
fig1.suptitle("Position Error with Confidence", fontsize=16)
ax[0].set_xlabel("Time step", fontsize=12)
ax[0].set_ylabel("Position error (km)", fontsize=16)
ax[1].set_xlabel("Time step", fontsize=12)
ax[1].set_ylabel("Position error (km)", fontsize=16)
ax[2].set_xlabel("Time step", fontsize=12)
ax[2].set_ylabel("Position error (km)", fontsize=16)
fig1.savefig(os.path.join(DIR_NAME, "position_error_with_confidence.png"))

# 2. EKF Position Error
fig2 = plt.figure()
plt.plot(ERROR[:, 0:3])
plt.legend(["x", "y", "z"])
plt.xlabel("Time step")
plt.ylabel("Position error [km]")
plt.title("EKF Position Error")
fig2.savefig(os.path.join(DIR_NAME, "ekf_position_error.png"))

# 3. EKF Position Error with Confidence
fig3 = plt.figure()
plt.plot(ERROR[:, 0:3])
plt.plot(3 * EKF_STATE_STD[:, 0:3], "r--")
plt.plot(-3 * EKF_STATE_STD[:, 0:3], "r--")
plt.legend(["x", "y", "z"])
plt.xlabel("Time step")
plt.ylabel("Position error [km]")
plt.title("EKF Position Error")
fig3.savefig(os.path.join(DIR_NAME, "ekf_position_error_with_confidence.png"))

# 4. EKF Velocity Error
fig4 = plt.figure()
plt.plot(ERROR[:, 3:6])
plt.legend(["x", "y", "z"])
plt.xlabel("Time step")
plt.ylabel("Velocity error [km/s]")
plt.title("EKF Velocity Error")
fig4.savefig(os.path.join(DIR_NAME, "ekf_velocity_error.png"))

# 5. EKF Unmodelled Acceleration
fig5 = plt.figure()
plt.plot(ERROR[:, 6:9])
plt.legend(["x", "y", "z"])
plt.xlabel("Time step")
plt.ylabel("Unmodelled acc error [km/s^2]")
plt.title("EKF Unmodelled Acceleration")
fig5.savefig(os.path.join(DIR_NAME, "ekf_unmodelled_acceleration.png"))

# 6. EKF Gyro Bias Error
fig6 = plt.figure()
plt.plot(ERROR[:, 17:])
plt.legend(["x", "y", "z"])
plt.xlabel("Time step")
plt.ylabel("Gyro bias error [rad/s]")
plt.title("EKF Gyro Bias Error")
fig6.savefig(os.path.join(DIR_NAME, "ekf_gyro_bias_error.png"))

# 7. EKF Covariance Trace
fig7 = plt.figure()
plt.plot(COV_TRACE)
plt.xlabel("Time step")
plt.ylabel("Covariance trace")
plt.title("EKF Covariance Trace")
fig7.savefig(os.path.join(DIR_NAME, "ekf_covariance_trace.png"))

# 8. EKF Covariance Condition Number
fig8_cond = plt.figure()
plt.plot(COV_COND_NUM)
plt.xlabel("Time step")
plt.ylabel("Covariance condition number")
plt.title("EKF Covariance Condition Number")
fig8_cond.savefig(os.path.join(DIR_NAME, "ekf_covariance_condition_number.png"))

# 8. Actual Gyro Bias
fig8 = plt.figure()
plt.plot(TRUE_STATE[:, 17:])
plt.plot(EKF_STATE[:, 17:])
plt.legend(["x", "y", "z"])
plt.xlabel("Time step")
plt.ylabel("Gyro bias [rad/s]")
plt.title("Gyro Bias")
fig8.savefig(os.path.join(DIR_NAME, "actual_gyro_bias.png"))

# 9. EKF Drag Estimate
fig9 = plt.figure()
plt.plot(EKF_STATE)
plt.xlabel("Time step")
plt.ylabel("Drag estimate")
plt.title("EKF Drag Estimate")
fig9.savefig(os.path.join(DIR_NAME, "ekf_drag_estimate.png"))

plt.show()
