"""
Post-processing and visualization script for analyzing the results of an Extended Kalman Filter (EKF)
applied to orbit determination.

This script loads EKF state estimates, standard deviations, true states, and covariance metrics from
NumPy files in a specified results directory. It computes estimation errors for each state variable
(position, velocity, unmodeled acceleration, drag scalar, quaternion, angular velocity, and gyro bias),
including quaternion error as axis-angle. It generates plots for:
    - State estimates vs. ground truth
    - Estimation errors with 3-sigma confidence intervals
    - Error norms with confidence intervals
    - EKF covariance trace and condition number

All plots are saved as PNG files in the results directory for further analysis.
"""

# pylint: disable=E1136  # pylint/issues/9590
# remove later
# mypy: disable-error-code=index
# mypy: disable-error-code=name-defined
# mypy: disable-error-code=misc
# pylint: disable=import-error,invalid-name,undefined-variable
import os
from typing import List, cast

import matplotlib.pyplot as plt
import numpy as np

from utils.math_utils import left_q, q_2_rot, quat_conjugate

#
INDEX_TRIAL = 0
DIR_NAME = f"results/ekf_realdrag/trial_{INDEX_TRIAL}"
EKF_STATE = np.load(os.path.join(DIR_NAME, "ekf_state.npy"))
EKF_STATE_STD = np.load(os.path.join(DIR_NAME, "ekf_state_std.npy"))
TRUE_STATE = np.load(os.path.join(DIR_NAME, "true_state.npy"))
COV_TRACE = np.load(os.path.join(DIR_NAME, "cov_trace.npy"))
COV_COND_NUM = np.load(os.path.join(DIR_NAME, "cov_cond_num.npy"))

# IDX_DICT_ENTRY = Dict[str, Union[List[str], str, str, int, slice, slice, slice]]
# : Dict[str, IDX_DICT_ENTRY]
IDX_DICT = {
    "Position": {
        "y_lbls": ["x", "y", "z"],
        "y_units": "km",
        "y_units_err": "km",
        "idx_set": 0,
        "idx": slice(0, 3),
        "idx_std": slice(0, 3),
        "idx_err": slice(0, 3),
    },
    "Velocity": {
        "y_lbls": ["vx", "vy", "vz"],
        "y_units": "km/s",
        "y_units_err": "km/s",
        "idx_set": 1,
        "idx": slice(3, 6),
        "idx_std": slice(3, 6),
        "idx_err": slice(3, 6),
    },
    "Unmodelled Acceleration": {
        "y_lbls": ["ax", "ay", "az"],
        "y_units": "km/s^2",
        "y_units_err": "km/s^2",
        "idx_set": 2,
        "idx": slice(6, 9),
        "idx_std": slice(6, 9),
        "idx_err": slice(6, 9),
    },
    "Drag Scalar": {
        "y_lbls": ["drag"],
        "y_units": "-",
        "y_units_err": "-",
        "idx_set": 3,
        "idx": slice(9, 10),
        "idx_std": slice(9, 10),
        "idx_err": slice(9, 10),
    },
    "Quaternion": {
        "y_lbls": ["q0", "q1", "q2", "q3"],
        "y_units": "",
        "y_units_err": "rad",
        "idx_set": 4,
        "idx": slice(10, 14),
        "idx_std": slice(10, 13),
        "idx_err": slice(10, 13),
    },
    "Angular Velocity": {
        "y_lbls": ["wx", "wy", "wz"],
        "y_units": "rad/s",
        "y_units_err": "rad/s",
        "idx_set": 5,
        "idx": slice(14, 17),
        "idx_std": slice(13, 16),
        "idx_err": slice(13, 16),
    },
    "Gyro Bias": {
        "y_lbls": ["bx", "by", "bz"],
        "y_units": "rad/s",
        "y_units_err": "rad/s",
        "idx_set": 6,
        "idx": slice(17, 20),
        "idx_std": slice(13, 16),
        "idx_err": slice(16, 19),
    },
}

NT = EKF_STATE.shape[0]  # pylint: disable=E1101
NUM_ENTRIES = len(IDX_DICT)
ERROR = np.zeros((NT, 19))
ERROR = TRUE_STATE - EKF_STATE

ERROR_NORM = np.zeros((NT, NUM_ENTRIES))
ERROR_NORM_STD = np.zeros((NT, NUM_ENTRIES))

# TODO: for each set of states, a joint error plot should be computed
for key, val in IDX_DICT.items():
    idx = val["idx"]
    idx_err = val["idx_err"]
    idx_set = val["idx_set"]
    if key == "Quaternion":
        for i in range(NT):
            # Quaternion error
            q_true = TRUE_STATE[i, idx]
            q_est = EKF_STATE[i, idx]
            q_err = left_q(q_est) @ quat_conjugate(q_true)
            if q_err[0] < 0:
                q_err = -q_err
            # convert to axis angle
            aa_err = q_2_rot(q_err)
            ERROR[i, idx_err] = aa_err
    else:
        ERROR[:, idx_err] = TRUE_STATE[:, idx] - EKF_STATE[:, idx]  # Drag scalar error
    if (idx_err.stop - idx_err.start) > 1:  # type: ignore
        ERROR_NORM[:, idx_set] = np.linalg.norm(ERROR[:, idx_err], axis=1)
        # TODO: Fix this to use the correct standard deviation for the l2-norm of gaussian samples
        ERROR_NORM_STD[:, idx_set] = np.linalg.norm(EKF_STATE_STD[:, idx_err], axis=1)


# Plot errors for each state variable in IDX_DICT
for key, val in IDX_DICT.items():
    idx = val["idx"]
    idx_std = val["idx_std"]
    idx_err = val["idx_err"]
    idx_set = val["idx_set"]
    y_lbls = cast(List[str], val["y_lbls"])
    y_units = val["y_units"]
    y_units_err = val["y_units_err"]
    # Handle slice (multiple variables) or int (single variable)
    if (idx.stop - idx.start) > 1:  # type: ignore
        num_entries = idx.stop - idx.start  # type: ignore
        # State plots
        fig, axes = plt.subplots(num_entries, 1, figsize=(8, 3 * num_entries), sharex=True)
        if num_entries == 1:
            axes = [axes]
        for i in range(num_entries):
            axes[i].plot(TRUE_STATE[:, idx.start + i], label="True")  # type: ignore
            axes[i].plot(EKF_STATE[:, idx.start + i], label="EKF")  # type: ignore
            axes[i].set_ylabel(f"{y_lbls[i]} [{y_units}]")
            axes[i].set_title(f"{key} {y_lbls[i]}")
            if i == num_entries - 1:
                axes[i].legend()
        axes[-1].set_xlabel("Time step")
        fig.tight_layout()
        fig.savefig(os.path.join(DIR_NAME, f"{key.lower().replace(' ', '_')}.png"))
        plt.close(fig)

        # EKF error plots
        num_entries = idx_err.stop - idx_err.start  # type: ignore
        fig, axes = plt.subplots(num_entries, 1, figsize=(8, 3 * num_entries), sharex=True)
        if num_entries == 1:
            axes = [axes]
        for i in range(num_entries):
            axes[i].plot(ERROR[:, idx_err.start + i], label="Error")  # type: ignore
            if idx_std is not None:
                axes[i].plot(3 * EKF_STATE_STD[:, idx_std.start + i], "r--", label="3σ")  # type: ignore
                axes[i].plot(-3 * EKF_STATE_STD[:, idx_std.start + i], "r--")  # type: ignore
            if i == num_entries - 1:
                axes[i].legend()
            axes[i].set_ylabel(f"{y_lbls[i]} [{y_units_err}]")
            axes[i].set_title(f"EKF {key} {y_lbls[i]} Error")
        axes[-1].set_xlabel("Time step")
        fig.tight_layout()
        fig.savefig(os.path.join(DIR_NAME, f"ekf_{key.lower().replace(' ', '_')}_error.png"))
        plt.close(fig)

        # Error norm plot
        fig_norm = plt.figure()
        plt.plot(ERROR_NORM[:, idx_set], label=f"{key} Error Norm")
        plt.fill_between(
            np.arange(NT),
            np.zeros(NT),
            3 * ERROR_NORM_STD[:, idx_set],
            color="red",
            alpha=0.2,
            label="3σ Confidence Interval",
        )
        plt.xlabel("Time step")
        plt.ylabel(f"{key} Error Norm [{y_units_err}]")
        plt.title(f"EKF {key} Error Norm")
        plt.legend()
        fig_norm.savefig(
            os.path.join(DIR_NAME, f"ekf_{key.lower().replace(' ', '_')}_error_norm.png")
        )
        plt.close(fig_norm)
    else:
        # State plots
        fig, ax = plt.subplots()
        ax.plot(TRUE_STATE[:, idx], label="True")
        ax.plot(EKF_STATE[:, idx], label="EKF")
        ax.set_xlabel("Time step")
        ax.set_ylabel(f"{y_lbls[0]} [{y_units}]")
        ax.set_title(f"{key} {y_lbls[0]}")
        ax.legend()
        fig.savefig(os.path.join(DIR_NAME, f"{key.lower().replace(' ', '_')}.png"))
        plt.close(fig)

        # EKF error plots
        fig, ax = plt.subplots()
        ax.plot(ERROR[:, idx_err], label="Error")
        if idx_std is not None:
            ax.plot(3 * EKF_STATE_STD[:, idx_std], "r--", label="3σ")
            ax.plot(-3 * EKF_STATE_STD[:, idx_std], "r--")
        ax.legend()
        ax.set_xlabel("Time step")
        ax.set_ylabel(f"{y_lbls[0]} error [{y_units_err}]")
        ax.set_title(f"EKF {key} {y_lbls[0]} Error")
        fig.savefig(os.path.join(DIR_NAME, f"ekf_{key.lower().replace(' ', '_')}_error.png"))
        plt.close(fig)


# 7. EKF Covariance Trace
fig7 = plt.figure()
plt.plot(COV_TRACE)
plt.xlabel("Time step")
plt.ylabel("Covariance trace")
plt.title("EKF Covariance Trace")
fig7.savefig(os.path.join(DIR_NAME, "ekf_covariance_trace.png"))
plt.close(fig)

# 8. EKF Covariance Condition Number
fig8_cond = plt.figure()
plt.plot(COV_COND_NUM)
plt.xlabel("Time step")
plt.ylabel("Covariance condition number")
plt.title("EKF Covariance Condition Number")
fig8_cond.savefig(os.path.join(DIR_NAME, "ekf_covariance_condition_number.png"))
plt.close(fig)
# plt.show()
