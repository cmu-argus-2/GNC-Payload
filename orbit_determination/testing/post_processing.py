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

import os

import matplotlib.pyplot as plt
import numpy as np

INDEX_TRIAL = 0
DIR_NAME = f"output_dir_ekf_realdrag/trial_{INDEX_TRIAL}"
ERROR = np.load(os.path.join(DIR_NAME, "pos_error.npy"))
VEL_ERROR = np.load(os.path.join(DIR_NAME, "vel_error.npy"))
SIGMA_HIGH = np.load(os.path.join(DIR_NAME, "sigma_high.npy"))
SIGMA_LOW = np.load(os.path.join(DIR_NAME, "sigma_low.npy"))
UA_ERROR = np.load(os.path.join(DIR_NAME, "ua_error.npy"))
GYRO_BIAS_ERROR = np.load(os.path.join(DIR_NAME, "gyro_bias_error.npy"))
ACTUAL_BIAS = np.load(os.path.join(DIR_NAME, "actual_bias.npy"))
DRAG_ESTIMATE = np.load(os.path.join(DIR_NAME, "drag_estimate.npy"))
COV_TRACE = np.load(os.path.join(DIR_NAME, "cov_trace.npy"))

# 1. Position Error with Confidence
fig1, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(ERROR[:, 0], label="x")  # pylint: disable=E1136  # pylint/issues/9590
ax[1].plot(ERROR[:, 1], label="y")  # pylint: disable=E1136  # pylint/issues/9590
ax[2].plot(ERROR[:, 2], label="z")  # pylint: disable=E1136  # pylint/issues/9590
ax[0].plot(SIGMA_HIGH[:, 0], "r--")  # pylint: disable=E1136  # pylint/issues/9590
ax[0].plot(SIGMA_LOW[:, 0], "r--")  # pylint: disable=E1136  # pylint/issues/9590
ax[1].plot(SIGMA_HIGH[:, 1], "r--")  # pylint: disable=E1136  # pylint/issues/9590
ax[1].plot(SIGMA_LOW[:, 1], "r--")  # pylint: disable=E1136  # pylint/issues/9590
ax[2].plot(SIGMA_HIGH[:, 2], "r--")  # pylint: disable=E1136  # pylint/issues/9590
ax[2].plot(SIGMA_LOW[:, 2], "r--")  # pylint: disable=E1136  # pylint/issues/9590
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
plt.plot(ERROR)
plt.legend(["x", "y", "z"])
plt.xlabel("Time step")
plt.ylabel("Position error [km]")
plt.title("EKF Position Error")
fig2.savefig(os.path.join(DIR_NAME, "ekf_position_error.png"))

# 3. EKF Position Error with Confidence
fig3 = plt.figure()
plt.plot(ERROR)
plt.plot(SIGMA_HIGH, "r--")
plt.plot(SIGMA_LOW, "r--")
plt.legend(["x", "y", "z"])
plt.xlabel("Time step")
plt.ylabel("Position error [km]")
plt.title("EKF Position Error")
fig3.savefig(os.path.join(DIR_NAME, "ekf_position_error_with_confidence.png"))

# 4. EKF Velocity Error
fig4 = plt.figure()
plt.plot(VEL_ERROR)
plt.legend(["x", "y", "z"])
plt.xlabel("Time step")
plt.ylabel("Velocity error [km/s]")
plt.title("EKF Velocity Error")
fig4.savefig(os.path.join(DIR_NAME, "ekf_velocity_error.png"))

# 5. EKF Unmodelled Acceleration
fig5 = plt.figure()
plt.plot(UA_ERROR)
plt.legend(["x", "y", "z"])
plt.xlabel("Time step")
plt.ylabel("Unmodelled acc error [km/s^2]")
plt.title("EKF Unmodelled Acceleration")
fig5.savefig(os.path.join(DIR_NAME, "ekf_unmodelled_acceleration.png"))

# 6. EKF Gyro Bias Error
fig6 = plt.figure()
plt.plot(GYRO_BIAS_ERROR)
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

# 8. Actual Gyro Bias
fig8 = plt.figure()
plt.plot(ACTUAL_BIAS)
plt.legend(["x", "y", "z"])
plt.xlabel("Time step")
plt.ylabel("Actual gyro bias [rad/s]")
plt.title("Actual Gyro Bias")
fig8.savefig(os.path.join(DIR_NAME, "actual_gyro_bias.png"))

# 9. EKF Drag Estimate
fig9 = plt.figure()
plt.plot(DRAG_ESTIMATE)
plt.xlabel("Time step")
plt.ylabel("Drag estimate")
plt.title("EKF Drag Estimate")
fig9.savefig(os.path.join(DIR_NAME, "ekf_drag_estimate.png"))

plt.show()
