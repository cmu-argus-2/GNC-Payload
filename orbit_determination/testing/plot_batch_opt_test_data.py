import datetime
import os

import brahe
import matplotlib.pyplot as plt
import numpy as np
import PIL
from dynamics.orbital_att_dynamics import DynamicsIDX as dynidx
from mpl_toolkits.mplot3d import Axes3D

from utils.earth_utils import ecef_to_lat_lon
from utils.plotting_utils import load_equirectangular_map


def plot_position(time, states: np.ndarray, dir_name: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, states[:, dynidx.RX], label="X")
    ax.plot(time, states[:, dynidx.RY], label="Y")
    ax.plot(time, states[:, dynidx.RZ], label="Z")
    ax.set_title("Position (km)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (km)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    if dir_name:
        fig.savefig(os.path.join(dir_name, "position.png"))


def plot_velocity(time, states: np.ndarray, dir_name: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, states[:, dynidx.VX], label="Vx")
    ax.plot(time, states[:, dynidx.VY], label="Vy")
    ax.plot(time, states[:, dynidx.VZ], label="Vz")
    ax.set_title("Velocity (km/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (km/s)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    if dir_name:
        fig.savefig(os.path.join(dir_name, "velocity.png"))


def plot_attitude(time, states: np.ndarray, dir_name: str):
    # Quaternion figure
    fig_q = plt.figure(figsize=(10, 4))
    ax_q = fig_q.add_subplot(111)
    ax_q.plot(time, states[:, dynidx.QW], label="Qw")
    ax_q.plot(time, states[:, dynidx.QX], label="Qx")
    ax_q.plot(time, states[:, dynidx.QY], label="Qy")
    ax_q.plot(time, states[:, dynidx.QZ], label="Qz")
    ax_q.set_title("Quaternion")
    ax_q.set_xlabel("Time (s)")
    ax_q.set_ylabel("Quaternion Components")
    ax_q.legend()
    ax_q.grid(True)
    fig_q.tight_layout()
    if dir_name:
        fig_q.savefig(os.path.join(dir_name, "quaternion.png"))


def plot_angular_velocity(time, states: np.ndarray, dir_name: str):
    fig_omega = plt.figure(figsize=(10, 4))
    ax_omega = fig_omega.add_subplot(111)
    ax_omega.plot(time, states[:, dynidx.OMEGA_X], label="Omega X")
    ax_omega.plot(time, states[:, dynidx.OMEGA_Y], label="Omega Y")
    ax_omega.plot(time, states[:, dynidx.OMEGA_Z], label="Omega Z")
    ax_omega.set_title("Angular Velocity (rad/s)")
    ax_omega.set_xlabel("Time (s)")
    ax_omega.set_ylabel("Angular Velocity (rad/s)")
    ax_omega.legend()
    ax_omega.grid(True)
    fig_omega.tight_layout()
    if dir_name:
        fig_omega.savefig(os.path.join(dir_name, "angular_velocity.png"))


# plot trajectory on map
def plot_trajectory_on_map(time, states: np.ndarray, dir_name: str):
    # Placeholder for actual implementation
    fig, ax = plt.subplots()
    # input time in unix seconds and states in eci
    lat = np.zeros(len(time))  # placeholder
    lon = np.zeros(len(time))  # placeholder
    eci_pos = states[:, 0:3]  # assuming first three columns are position in ECI
    x_ecef_list = np.zeros((len(time), 3))
    # unix time to epoch
    for i in range(len(time)):
        datei = datetime.datetime.fromtimestamp(time[i])
        epoch = brahe.Epoch(datei)
        x_eci = eci_pos[i, :]  # ECI position at this time

        x_ecef = brahe.frames.rECItoECEF(epc=epoch) @ x_eci
        x_ecef_list[i, :] = x_ecef

    # convert to lat lon
    lat_lon = ecef_to_lat_lon(x_ecef_list)
    lat = lat_lon[:, 0]
    lon = lat_lon[:, 1]
    file_location = os.path.join(
        os.path.dirname(__file__), "../../utils/equirectangular_map.png"
    )
    ax.imshow(load_equirectangular_map(file_location), extent=(-180, 180, -90, 90))
    ax.plot(lon, lat, color="red", linewidth=1)
    ax.set_title("Ground Track")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    fig.tight_layout()
    if dir_name:
        fig.savefig(os.path.join(dir_name, "trajectory_on_map.png"))


# plot trajectory in ecef with landmark measurements and visibility cones
def plot_trajectory_in_ecef(time, states: np.ndarray, ld_meas, dir_name: str):
    eci_pos = states[:, 0:3]  # assuming first three columns are position in ECI
    x_ecef_list = np.zeros((len(time), 3))
    # unix time to epoch
    for i in range(len(time)):
        datei = datetime.datetime.fromtimestamp(time[i])
        epoch = brahe.Epoch(datei)
        x_eci = eci_pos[i, :]  # ECI position at this time

        x_ecef = brahe.frames.rECItoECEF(epc=epoch) @ x_eci
        x_ecef_list[i, :] = x_ecef

    x_mean = np.mean(x_ecef_list, axis=0)
    x_mean /= np.linalg.norm(x_mean)
    azim = np.rad2deg(np.arctan2(x_mean[1], x_mean[0])) + 45
    elev = np.rad2deg(np.arcsin(x_mean[2]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    # earth 3d - from https://stackoverflow.com/questions/30269099/how-to-plot-a-rotating-3d-earth
    # load bluemarble with PIL
    file_location = os.path.join(
        os.path.dirname(__file__), "../../utils/equirectangular_map.png"
    )
    bm = PIL.Image.open(file_location)
    n = 1
    bm = np.array(bm.resize([d // 1 for d in bm.size])) / 256
    lons = np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
    lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180

    # plot Earth for context
    x = np.outer(np.cos(lons), 6378 * np.cos(lats)).T
    y = np.outer(np.sin(lons), 6378 * np.cos(lats)).T
    z = np.outer(np.ones(np.size(lons)), 6356 * np.sin(lats)).T
    ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors=bm, zorder=1)

    # plot satellite trajectory
    ax.plot(
        x_ecef_list[:, 0],
        x_ecef_list[:, 1],
        x_ecef_list[:, 2],
        color="red",
        linewidth=1.5,
        label="Satellite Trajectory",
        zorder=2,
    )

    ax.set_title("Trajectory in ECEF with Landmark Measurements and Visibility Lines")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.view_init(azim=azim, elev=elev)
    lim = 8000
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    if dir_name:
        fig.savefig(os.path.join(dir_name, "trajectory_in_ecef.png"))

    # plotting a zoom-in of the trajectory


# plot gyro bias
def plot_gyro_bias(time, states: np.ndarray, dir_name: str):
    # Placeholder for actual implementation
    gyro_bias = states[:, 13:16]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, gyro_bias[:, 0], label="Bias X")
    ax.plot(time, gyro_bias[:, 1], label="Bias Y")
    ax.plot(time, gyro_bias[:, 2], label="Bias Z")
    ax.set_title("Gyro Bias")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Gyro Bias (rad/s)")
    ax.grid(True)
    fig.tight_layout()
    if dir_name:
        fig.savefig(os.path.join(dir_name, "gyro_bias.png"))


# plot gyro measurements
def plot_gyro_measurements(time, gyro_meas: np.ndarray, dir_name: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gyro_meas[:, 0], gyro_meas[:, 1], label="Gyro X")
    ax.plot(gyro_meas[:, 0], gyro_meas[:, 2], label="Gyro Y")
    ax.plot(gyro_meas[:, 0], gyro_meas[:, 3], label="Gyro Z")
    ax.set_title("Gyro Measurements")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    if dir_name:
        fig.savefig(os.path.join(dir_name, "gyro_measurements.png"))


def plot_landmark_measurements(time, ld_meas, dir_name: str):
    fig, ax = plt.subplots(3, 2, figsize=(8, 4.5))
    ax[0, 0].plot(ld_meas[:, 0], ld_meas[:, 1], marker="o", linestyle="", markersize=4)
    ax[1, 0].plot(ld_meas[:, 0], ld_meas[:, 2], marker="o", linestyle="", markersize=4)
    ax[2, 0].plot(ld_meas[:, 0], ld_meas[:, 3], marker="o", linestyle="", markersize=4)
    ax[0, 1].plot(ld_meas[:, 0], ld_meas[:, 4], marker="o", linestyle="", markersize=4)
    ax[1, 1].plot(ld_meas[:, 0], ld_meas[:, 5], marker="o", linestyle="", markersize=4)
    ax[2, 1].plot(ld_meas[:, 0], ld_meas[:, 6], marker="o", linestyle="", markersize=4)
    fig.suptitle("Landmark measurements")
    ax[2, 0].set_xlabel("Time (s)")
    ax[2, 1].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("X bear (-)")
    ax[1, 0].set_ylabel("Y bear (-)")
    ax[2, 0].set_ylabel("Z bear (-)")
    ax[0, 1].set_ylabel("X ld (km)")
    ax[1, 1].set_ylabel("Y ld (km)")
    ax[2, 1].set_ylabel("Z ld (km)")
    for i in range(3):
        for j in range(2):
            ax[i, j].grid(True)
    fig.tight_layout()
    if dir_name:
        fig.savefig(os.path.join(dir_name, "landmark_measurements.png"))


# main plotting function
def plot_syn_data(time, states: np.ndarray, ld_meas, gyro_meas, dir_name: str):
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Position figure
    plot_position(time, states, dir_name)

    # Velocity figure
    plot_velocity(time, states, dir_name)

    # Quaternion figure
    plot_attitude(time, states, dir_name)

    # Angular velocity figure
    plot_angular_velocity(time, states, dir_name)

    # plot trajectory on map
    plot_trajectory_on_map(time, states, dir_name)

    # plot trajectory in ecef with landmark measurements and visibility cones
    plot_trajectory_in_ecef(time, states, ld_meas, dir_name)

    # plot gyro bias
    plot_gyro_bias(time, states, dir_name)

    # plot gyro measurements
    plot_gyro_measurements(time, gyro_meas, dir_name)

    # plot landmark measurements
    plot_landmark_measurements(time, ld_meas, dir_name)
