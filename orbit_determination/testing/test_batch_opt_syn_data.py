import datetime
import os

import brahe
import matplotlib.pyplot as plt
import numpy as np
from utils.math_utils import quat2rotm

def test_landmark_measurements(time, ld_meas, states):
    # do triad on the measurements to check if we get accurate attitude estimates
    landmark_times = ld_meas[:, 0]
    groups = [1]
    for i in range(1,len(landmark_times)):
        if landmark_times[i] != landmark_times[i-1]:
            groups.append(1)
        else:
            groups[-1] += 1
            
    print(groups)
    state_group_ids = []
    for i in range(len(groups)):
        state_group_ids.append(np.argmin(abs(time - landmark_times[int(np.sum(groups[:i]))])))
    # state_ld_times = time[state_group_ids]
    states_ls = states[state_group_ids]
    attitudes = states_ls[:,6:10]
    positions = states_ls[:,0:3]
    for i, _ in enumerate(groups):
        n_meas = groups[i]
        if n_meas < 2:
            continue
        if i == 0:
            idvec = range(0, groups[i])
        else:
            idvec = range(np.sum(groups[:i]), np.sum(groups[:i+1]))
        meas = ld_meas[idvec,1:4]
        landmarks = ld_meas[idvec,4:7] - positions[i]
        
        R_b2i = quat2rotm(attitudes[i])
        
        est_R = triad(meas[:2,:].T, landmarks[:2,:].T)
        R_diff  = est_R @ R_b2i.T
        error_angle = np.arccos((np.trace(R_diff) - 1) / 2)
        
        ldmks_b = landmarks @ R_b2i
        
        # vector of angles between landmarks and measurements
        ldmks_b_norm = ldmks_b / np.linalg.norm(ldmks_b, axis=1, keepdims=True)
        meas_norm = meas / np.linalg.norm(meas, axis=1, keepdims=True)
        angles = np.arccos(np.clip(np.sum(ldmks_b_norm * meas_norm, axis=1), -1, 1))
        
        # est_attitude = brahe.attitude.triad(landmarks, meas, positions[i], R_b2i)
        # est_quat = brahe.utils.rotmat_to_quat(est_attitude)
        # error_angle = brahe.attitude.quat_angle_diff(est_quat, attitudes[i])
        print(f"Landmark measurement at time {landmark_times[int(np.sum(groups[:i]))]} with {n_meas} measurements")
        print(f"Triad error (deg): {np.degrees(error_angle)}")
        print(f"Mean measurement angle (deg): {np.degrees(np.mean(angles))}")
        print(f"Max measurement angle (deg): {np.degrees(np.max(angles))}")


def triad(vecsb, vecsi):
    """
    TRIAD algorithm to compute the rotation matrix from two vector observations.

    Args:
        vecsb (np.ndarray): 3x2 matrix of body-frame
                            observations (each column is a vector).
        vecsi (np.ndarray): 3x2 matrix of inertial-frame
                            observations (each column is a vector). 
    Returns:
        np.ndarray: 3x3 rotation matrix from body to inertial frame.
    """
    # Normalize input vectors
    t1b = vecsb[:, 0] / np.linalg.norm(vecsb[:, 0])
    t2b = vecsb[:, 1] / np.linalg.norm(vecsb[:, 1])
    t1i = vecsi[:, 0] / np.linalg.norm(vecsi[:, 0])
    t2i = vecsi[:, 1] / np.linalg.norm(vecsi[:, 1])

    # Compute orthonormal bases
    t3b = np.cross(t1b, t2b)
    t3b = t3b / np.linalg.norm(t3b)
    t2b = np.cross(t3b, t1b)
    t2b = t2b / np.linalg.norm(t2b)
    t3i = np.cross(t1i, t2i)
    t3i = t3i / np.linalg.norm(t3i)
    t2i = np.cross(t3i, t1i)
    t2i = t2i / np.linalg.norm(t2i)

    # Form rotation matrices
    Tb = np.column_stack((t1b, t2b, t3b))
    Ti = np.column_stack((t1i, t2i, t3i))

    # Compute rotation matrix from body to inertial frame
    R_b2i = Ti @ Tb.T

    return R_b2i    

# main plotting function
def test_syn_data(time, states: np.ndarray, ld_meas, gyro_meas):

    # plot landmark measurements
    test_landmark_measurements(time, ld_meas, states)
