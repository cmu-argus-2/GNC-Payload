"""
Generate batch-optimisation training data from simulated SSO trajectories.

Usage:
    python -m orbit_determination.test_batch_opt_data_gen [options]

Options:
    --altitude FLOAT    Orbit altitude in metres          (default 590000)
    --ltan FLOAT        Local time of ascending node, h  (default 11.0)
    --mgrs STR          MGRS GZD for initial position      (default 17R)
    --nadir-att         Nadir-pointing initial attitude   (default)
    --random-att        Random initial attitude
    --nadir-rate        Nadir-pointing angular velocity   (default)
    --random-rate       Random angular velocity
    --rate-std FLOAT    Random rate std dev, dps          (default 1.0)
    --trials INT        Number of trials                  (default 1)
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import brahe
import cv2
import h5py
import numpy as np
import quaternion
from brahe.epoch import Epoch

from orbit_determination.landmark_bearing_sensors import SimulatedMLLandmarkBearingSensor
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from orbit_determination.testing.plot_batch_opt_test_data import plot_syn_data
from orbit_determination.testing.test_batch_opt_syn_data import test_syn_data
from simulation.dynamics.orbital_att_dynamics import Dynamics
from simulation.dynamics.orbital_att_dynamics import DynamicsIDX as dynidx
from simulation.sensors.camera_model import CameraModel, CameraModelManager
from utils.brahe_utils import load_brahe_data_files_if_needed
from utils.config_utils import load_config
from utils.earth_utils import get_nadir_rotation
from utils.orbit_utils import get_cos_sso_inclination, get_sso_orbit_state

# pylint: disable=too-many-locals,too-many-statements

_MGRS_BANDS = "CDEFGHJKLMNPQRSTUVWX"

# Hardware camera-id mapping: x+→0, y+→1, x-→2, y-→3
_CAM_NAME_TO_ID = {"x+": 0, "y+": 1, "y-": 2, "x-": 3}
_CAM_NAME_TO_STR = {"x+": "xp", "y+": "yp", "x-": "xm", "y-": "ym"}

_MGRS_GZD_NAMES = {
    # "10S": "California",
    # "10T": "Oregon",
    # "11R": "Baja",
    # "12R": "Sonora",
    # "16T": "Minnesota",
    "17R": "Florida",
    "17T": "Toronto",
    # "18S": "NewJersey",
    # "32S": "Tunisia",
    # "32T": "Switzerland",
    # "33S": "Sicilia",
    # "33T": "Italy",
    # "52S": "Korea",
    # "53S": "Hiroshima",
    # "54S": "Tokyo",
    # "54T": "Sapporo",
}


# ---------------------------------------------------------------------------
# MGRS helpers
# ---------------------------------------------------------------------------

def _mgrs_gzd_bounds(gzd: str):
    """Return (lat_min, lat_max, lon_min, lon_max) for an MGRS GZD like '17R'."""
    zone_num = int(gzd[:-1])
    band_idx = _MGRS_BANDS.index(gzd[-1].upper())
    lat_min = -80 + band_idx * 8
    lat_max = lat_min + 8
    lon_min = (zone_num - 1) * 6 - 180
    lon_max = lon_min + 6
    return lat_min, lat_max, lon_min, lon_max


def _sub_satellite_lat_lon(epoch: Epoch, state_km: np.ndarray):
    pos_ecef = brahe.frames.rECItoECEF(epoch) @ (state_km[:3] * 1e3)
    lat = np.degrees(np.arcsin(pos_ecef[2] / np.linalg.norm(pos_ecef)))
    lon = np.degrees(np.arctan2(pos_ecef[1], pos_ecef[0]))
    return lat, lon


def _is_over_mgrs_gzd(epoch: Epoch, state_km: np.ndarray, gzd: str) -> bool:
    lat_min, lat_max, lon_min, lon_max = _mgrs_gzd_bounds(gzd)
    lat, lon = _sub_satellite_lat_lon(epoch, state_km)
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


# ---------------------------------------------------------------------------
# Orbit / attitude helpers
# ---------------------------------------------------------------------------

def _ascending_node_lon(epoch: Epoch, ltan_h: float) -> float:
    """ECEF longitude [deg] of the ascending node for the desired LTAN."""
    sun_eci = brahe.ephemerides.sun_position(epoch)
    sun_ecef = brahe.frames.rECItoECEF(epoch) @ sun_eci
    sun_lon = np.degrees(np.arctan2(sun_ecef[1], sun_ecef[0]))
    return ((sun_lon + (ltan_h - 12.0) * 15.0 + 180.0) % 360.0) - 180.0

# TODO: fix 
def _nadir_rate(state_km: np.ndarray) -> np.ndarray:
    """Body-frame angular velocity [rad/s] for nadir pointing.

    get_nadir_rotation sets z_body = orbit-normal (h_hat), so the angular
    velocity needed to maintain nadir pointing is n * z_body = [0, 0, n].
    """
    a_m = np.linalg.norm(state_km[:3]) * 1e3
    n = np.sqrt(brahe.GM_EARTH / a_m ** 3)
    return np.array([0.0, 0.0, n])


def _orbit_state_over_mgrs(
    epoch: Epoch, altitude_m: float, ltan_h: float, gzd: str, northwards: bool = True
) -> np.ndarray:
    """
    Return SSO state [km, km/s] for a satellite positioned above the centre of
    the given MGRS GZD, on the orbit plane defined by ltan_h.

    The formula places the satellite at the MGRS centre latitude and computes
    the ECEF longitude analytically from the LTAN-derived ascending-node
    longitude and the SSO inclination, so the resulting orbit has the requested
    LTAN without needing propagation.
    """
    lat_min, lat_max, lon_min, lon_max = _mgrs_gzd_bounds(gzd)
    lat_c = lat_min
    lon_c = (lon_min + lon_max) / 2.0

    state_m = get_sso_orbit_state(epoch, lat_c, lon_c, altitude_m, northwards=northwards)
    return state_m / 1e3  # m, m/s → km, km/s


def _epoch_from_unix(t_unix: float) -> Epoch:
    dt = datetime.fromtimestamp(t_unix, tz=timezone.utc)
    return Epoch(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                 dt.second + dt.microsecond / 1e6)


def _find_epoch_over_mgrs(
    config_start_mjd: float,
    altitude_m: float,
    ltan_h: float,
    gzd: str,
    northwards: bool = True,
) -> Epoch:
    """
    Return the epoch closest to config_start_mjd at which an SSO with the
    given LTAN would pass over the centre of the MGRS GZD.

    For a satellite at MGRS centre longitude lon_c on an LTAN-defined orbit:
        an_lon = sun_lon + (ltan_h - 12) * 15
        lon_c  = an_lon + dlon  →  sun_lon = lon_c - dlon - (ltan_h-12)*15

    We find the epoch where the sub-solar ECEF longitude equals that target,
    using Newton iterations (converges in 2-3 steps to < 0.1° accuracy).
    """
    lat_min, lat_max, lon_min, lon_max = _mgrs_gzd_bounds(gzd)
    lat_c = lat_max # (lat_min + lat_max) / 2.0
    lon_c = (lon_min + lon_max) / 2.0

    i_sso = np.arccos(get_cos_sso_inclination(altitude_m))
    sin_u = np.clip(np.sin(np.radians(lat_c)) / np.sin(i_sso), -1.0, 1.0)
    u = np.arcsin(sin_u) if northwards else np.pi - np.arcsin(sin_u)
    dlon = np.degrees(np.arctan2(np.cos(i_sso) * np.sin(u), np.cos(u)))

    target_sun_lon = (lon_c - dlon - (ltan_h - 12.0) * 15.0 + 360.0) % 360.0 - 180.0

    def _sun_lon(epoch: Epoch) -> float:
        sun_ecef = brahe.frames.rECItoECEF(epoch) @ brahe.ephemerides.sun_position(epoch)
        return np.degrees(np.arctan2(sun_ecef[1], sun_ecef[0]))

    t0_epoch = Epoch(*brahe.time.mjd_to_caldate(config_start_mjd))
    t0_unix = t0_epoch.to_datetime().replace(tzinfo=timezone.utc).timestamp()

    # Initial estimate: solar day ≈ 86400 s for 360°
    delta = (target_sun_lon - _sun_lon(t0_epoch) + 180.0) % 360.0 - 180.0
    dt_s = delta / (360.0 / 86400.0)

    # Newton refinement — converges in 2-3 iterations
    for _ in range(4):
        epoch_try = _epoch_from_unix(t0_unix + dt_s)
        residual = (target_sun_lon - _sun_lon(epoch_try) + 180.0) % 360.0 - 180.0
        dt_s += residual / (360.0 / 86400.0)

    epoch_found = _epoch_from_unix(t0_unix + dt_s)
    print(
        f"Starting epoch adjusted to {epoch_found} "
        f"(offset {dt_s/3600:+.2f} h from config date) "
        f"for LTAN={ltan_h}h over {gzd}"
    )
    return epoch_found


def _random_rotation() -> np.ndarray:
    rot = np.eye(3) + np.random.normal(0, 1e-3, (3, 3))
    rot = rot @ np.linalg.inv(np.linalg.cholesky(rot.T @ rot))
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-3)
    return rot


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _collect_measurements(data_manager, t, z, all_lm, all_gs):
    """Append non-empty landmark measurements to accumulators."""
    if z[0].shape[0] == 0:
        return all_lm, all_gs
    z1 = np.concatenate([z[0], z[1]], axis=1)
    tmp = np.full((z[0].shape[0], 1), t)
    lm = np.concatenate([tmp, z1], axis=1)
    gs = np.zeros(z[0].shape[0])
    gs[0] = 1
    return np.vstack([all_lm, lm]), np.concatenate([all_gs, gs])


# ---------------------------------------------------------------------------
# Hardware-format dataset writer
# ---------------------------------------------------------------------------

def _write_hardware_format_dataset(
    run_dir: str,
    run_ms: int,
    frame_records: list,
    all_gyro: np.ndarray,
    gyro_rate_hz: float,
    image_rate_hz: float,
    duration_s: float,
) -> None:
    """Write dataset.json, dataset_config.toml, imu_data.csv, and frame_*.json files.

    frame_records: list of (timestamp_ms, cam_id_int, cam_str_id) tuples.
    Renames camera-string JPG files to numeric cam-id naming when present.
    Falls back to PNG->JPG conversion for backward compatibility.
    """
    frame_id_list = []
    frame_json_entries = []
    dataset_folder = os.path.basename(run_dir)

    for ts_ms, cam_id_int, cam_str in frame_records:
        jpg_camstr_path = os.path.join(run_dir, f"raw_{ts_ms}_{cam_str}.jpg")
        png_path = os.path.join(run_dir, f"raw_{ts_ms}_{cam_str}.png")
        jpg_path = os.path.join(run_dir, f"raw_{ts_ms}_{cam_id_int}.jpg")

        width, height = CameraModel.IMAGE_WIDTH, CameraModel.IMAGE_HEIGHT
        size_bytes = 0

        if os.path.exists(jpg_camstr_path):
            if os.path.abspath(jpg_camstr_path) != os.path.abspath(jpg_path):
                os.replace(jpg_camstr_path, jpg_path)
            size_bytes = os.path.getsize(jpg_path)
        elif os.path.exists(png_path):
            img = cv2.imread(png_path)
            if img is not None:
                height, width = img.shape[:2]
                cv2.imwrite(jpg_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                size_bytes = os.path.getsize(jpg_path)
            os.remove(png_path)

        frame_id_list.append([cam_id_int, ts_ms])
        frame_json_entries.append({
            "timestamp": ts_ms,
            "cam_id": cam_id_int,
            "annotation_state": 0,
            "processing_stage": 0,
            "rank": 0.0,
            "inference_results": {
                "rcnet_version": -1,
                "ldnet_version": -1,
                "detected_regions_count": 0,
                "detected_landmarks_count": 0,
                "regions": [],
                "landmarks": [],
            },
            "raw_image": {
                "path": f"data/datasets/{dataset_folder}/raw_{ts_ms}_{cam_id_int}.jpg",
                "format": "JPG",
                "width": width,
                "height": height,
                "size_bytes": size_bytes,
            },
        })

    # Write per-frame JSON files
    for entry in frame_json_entries:
        ts_ms = entry["timestamp"]
        cam_id_int = entry["cam_id"]
        frame_path = os.path.join(run_dir, f"frame_{ts_ms}_{cam_id_int}.json")
        with open(frame_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent="\t")

    # Write dataset.json
    target_frame_nb = len(frame_records)

    dataset_json = {
        "capture_start_time": run_ms,
        "dataset_capture_mode": 2,
        "folder_path": f"data/datasets/{dataset_folder}/",
        "frame_id_list": frame_id_list,
        "frames_collected": len(frame_records),
        "image_capture_rate": 1.0 / image_rate_hz,
        "imu_collection_mode": 1,
        "imu_log_file_path": f"data/datasets/{dataset_folder}/imu_data.csv",
        "imu_sample_rate_hz": gyro_rate_hz,
        "imu_timestamps_collected": len(all_gyro),
        "maximum_period": duration_s,
        "num_frames_earth": 0,
        "num_frames_landmarks": 0,
        "num_frames_ldneted": 0,
        "num_frames_prefiltered": 0,
        "num_frames_rcneted": 0,
        "num_frames_roi": 0,
        "target_frame_nb": target_frame_nb,
        "target_processing_stage": 0,
    }
    with open(os.path.join(run_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent="\t")

    # Write dataset_config.toml
    toml_content = (
        f"capture_start_time = {run_ms}\n"
        f"dataset_capture_mode = 2\n"
        f"image_capture_rate = {1.0 / image_rate_hz}\n"
        f"imu_collection_mode = 1\n"
        f"imu_sample_rate_hz = {gyro_rate_hz}\n"
        f"maximum_period = {duration_s}\n"
        f"target_frame_nb = {target_frame_nb}\n"
        f"target_processing_stage = 0\n"
    )
    with open(os.path.join(run_dir, "dataset_config.toml"), "w", encoding="utf-8") as f:
        f.write(toml_content)

    # Write imu_data.csv in hardware format
    imu_csv_rows = np.column_stack([
        (all_gyro[:, 0] * 1000).astype(np.int64),
        np.degrees(all_gyro[:, 1:4]),
    ])
    np.savetxt(
        os.path.join(run_dir, "imu_data.csv"),
        imu_csv_rows,
        delimiter=",",
        header="Timestamp_ms, Gyro_X_dps, Gyro_Y_dps, Gyro_Z_dps",
        comments="",
        fmt=["%d", "%.6f", "%.6f", "%.6f"],
    )


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_simulation(
    run_dir: str,
    run_ms: int,
    altitude_m: float = 590e3,
    ltan_h: float = 11.0,
    mgrs_gzd: str = "17R",
    nadir_att: bool = True,
    nadir_rate: bool = True,
    rate_std_dps: float = 1.0,
    gyro_rate_hz: float = 10.0,
    image_rate_hz: float = 1 / 60.0,
    duration_s: float = 130.0,
    ld_version: int = 1,
    save_lat_lon: bool = False,
    write_labels: bool = True,
    use_gpu: bool = True,
    gpu_preload_region: str | None = None,
) -> None:
    idx = dynidx(has_gyro_bias=True)
    config = load_config()
    config["solver"]["world_update_rate"] = 2  # Hz
    config["mission"]["duration"] = duration_s

    dt = 1 / config["solver"]["world_update_rate"]
    starting_epoch = _find_epoch_over_mgrs(
        config["mission"]["start_date"], altitude_m, ltan_h, mgrs_gzd
    )
    N = int(np.ceil(duration_s / dt))
    epochs_list = starting_epoch.to_datetime().replace(tzinfo=timezone.utc).timestamp() + np.arange(N) * dt

    # --- SSO orbit: start over the MGRS GZD centre on the LTAN-defined trajectory ---
    initial_orb = _orbit_state_over_mgrs(starting_epoch, altitude_m, ltan_h, mgrs_gzd)

    # --- Initial attitude ---
    if nadir_att:
        # get_nadir_rotation: x+ points nadir, z_body = orbit-normal
        eci_R_body = get_nadir_rotation(initial_orb, nadir_axis="x-")
        init_quat = quaternion.from_rotation_matrix(eci_R_body)
    else:
        init_quat = quaternion.from_rotation_matrix(_random_rotation())

    # --- Initial angular velocity ---
    omega_init = _nadir_rate(initial_orb) if nadir_rate else \
        np.random.normal(0, np.radians(rate_std_dps), 3)

    initial_state = np.zeros(idx.NX)
    initial_state[idx.ORB] = initial_orb
    initial_state[idx.QUAT] = quaternion.as_float_array(init_quat)
    initial_state[idx.OMEGA] = omega_init
    initial_state[idx.GYR_BIAS] = np.random.normal(
        0, config["satellite"]["gyro"]["bias_std"], 3
    )

    landmark_bearing_sensor = SimulatedMLLandmarkBearingSensor(
        use_cesium=False, run_inference=False, ld_version=ld_version, mgrs_gzd=mgrs_gzd,
        save_lat_lon=save_lat_lon,
        write_labels=write_labels,
        use_gpu=use_gpu,
        preload_regions=[gpu_preload_region or mgrs_gzd] if use_gpu else None,
    )
    camera_model_manager = CameraModelManager()
    data_manager = ODSimulationDataManager(starting_epoch, dt, idx)
    data_manager.push_next_state(initial_state)

    ground_truth_dynamics = Dynamics(
        config=config,
        use_drag=True,
        use_j2=True,
        use_j34=True,
        use_sun_grav=True,
        use_moon_grav=True,
        include_gyro_bias=True,
        gyro_bias_tau=config["satellite"]["gyro"]["bias_tau"],
        gyro_bias_std=config["satellite"]["gyro"]["bias_std"],
    )

    imu_dt = 1.0 / gyro_rate_hz
    vis_dt = 1.0 / image_rate_hz
    last_imu = epochs_list[0]

    all_lm = np.zeros((0, 7))        # [t, bx, by, bz, lx, ly, lz]
    all_gs = np.array([])            # group-start flags
    all_gyro = np.zeros((0, 4))      # [t_unix_s, gx_rad, gy_rad, gz_rad]
    accel_components = {k: np.zeros((0, 3)) for k in
                        ["earth_gravity", "j2", "j34", "drag", "sun_gravity", "moon_gravity"]}
    frame_records = []               # (timestamp_ms, cam_id_int, cam_str_id)

    last_vis = -np.inf

    # --- Initial gyro measurement ---
    t0 = epochs_list[0]
    g0 = initial_state[idx.OMEGA] + initial_state[idx.GYR_BIAS]
    g0 += np.random.normal(0, config["satellite"]["gyro"]["noise_density"], 3)
    all_gyro = np.vstack([all_gyro, np.concatenate([[t0], g0])])

    a_comp = ground_truth_dynamics.get_accel_components(initial_state, epoch=starting_epoch)
    for k, v in a_comp.items():
        accel_components[k] = np.vstack([accel_components[k], v])

    # --- Initial measurement ---
    for cam in CameraModelManager.CAMERA_NAMES:
        data_manager.take_measurement(
            landmark_bearing_sensor, camera_model_manager[cam],
            idx=0, output_dir=run_dir,
        )
        ts_ms = int(data_manager.latest_epoch.to_datetime().replace(tzinfo=timezone.utc).timestamp() * 1000)
        frame_records.append((ts_ms, _CAM_NAME_TO_ID[cam], _CAM_NAME_TO_STR[cam]))
    _, *z = data_manager.latest_measurements
    all_lm, all_gs = _collect_measurements(data_manager, t0, z, all_lm, all_gs)
    if z[0].shape[0] > 0:
        print(f"Measurement at t={t0:.1f}s with {z[0].shape[0]} landmarks")
    last_vis = t0

    # --- Main loop ---
    for i in range(1, N):
        t = epochs_list[i]
        t_epc = data_manager.latest_epoch
        x = data_manager.latest_state

        next_state = ground_truth_dynamics.perturbed_f(x=x, dt=dt, epoch=t_epc)

        a_comp = ground_truth_dynamics.get_accel_components(next_state, epoch=t_epc)
        for k, v in a_comp.items():
            accel_components[k] = np.vstack([accel_components[k], v])

        data_manager.push_next_state(next_state)

        if last_imu + imu_dt <= t:
            g = next_state[idx.OMEGA] + next_state[idx.GYR_BIAS]
            g += np.random.normal(0, config["satellite"]["gyro"]["noise_density"], 3)
            all_gyro = np.vstack([all_gyro, np.concatenate([[t], g])])
            last_imu = t

        if last_vis + vis_dt <= t:
            last_vis = t  # always advance, regardless of landmark detections
            for cam in CameraModelManager.CAMERA_NAMES:
                data_manager.take_measurement(
                    landmark_bearing_sensor, camera_model_manager[cam],
                    idx=i, output_dir=run_dir,
                )
                ts_ms = int(data_manager.latest_epoch.to_datetime().replace(tzinfo=timezone.utc).timestamp() * 1000)
                frame_records.append((ts_ms, _CAM_NAME_TO_ID[cam], _CAM_NAME_TO_STR[cam]))
            print(f"Completion: {100 * i / N:.2f}%")
            _, *z = data_manager.latest_measurements
            all_lm, all_gs = _collect_measurements(data_manager, t, z, all_lm, all_gs)
            if z[0].shape[0] > 0:
                print(f"Measurement at t={t:.1f}s with {z[0].shape[0]} landmarks")

    # --- test_specific/ subfolder: truth data, plots, gyro params ---
    test_dir = os.path.join(run_dir, "test_specific")
    os.makedirs(test_dir, exist_ok=True)

    all_gs = np.expand_dims(all_gs, axis=1)

    with h5py.File(os.path.join(test_dir, "orbit_measurements.h5"), "w") as f:
        f.create_dataset("landmark_measurements", data=all_lm)
        f.create_dataset("gyro_measurements", data=all_gyro)
        f.create_dataset("group_starts", data=all_gs)

    with h5py.File(os.path.join(test_dir, "ground_truth_states.h5"), "w") as f:
        f.create_dataset("states", data=data_manager.states)
        f.create_dataset("unixtime", data=epochs_list)
        for k, v in accel_components.items():
            f.create_dataset(f"accel_{k}", data=v)

    with open(os.path.join(test_dir, "gyro_params.txt"), "w", encoding="utf-8") as f:
        f.write(f"gyro_noise_density={config['satellite']['gyro']['noise_density']}\n")
        f.write(f"gyro_bias_std={config['satellite']['gyro']['bias_std']}\n")
        f.write(f"gyro_bias_tau={config['satellite']['gyro']['bias_tau']}\n")

    with open(os.path.join(test_dir, "dynamic_params.txt"), "w", encoding="utf-8") as f:
        f.write(f"Cd={config['satellite']['Cd']}\n")
        f.write(f"area={config['satellite']['area']}\n")
        f.write(f"mass={config['satellite']['mass']}\n")

    plot_syn_data(
        epochs_list, data_manager.states, all_lm, all_gyro, accel_components, test_dir
    )

    # --- Hardware-format dataset files in run_dir ---
    _write_hardware_format_dataset(
        run_dir=run_dir,
        run_ms=run_ms,
        frame_records=frame_records,
        all_gyro=all_gyro,
        gyro_rate_hz=gyro_rate_hz,
        image_rate_hz=image_rate_hz,
        duration_s=duration_s,
    )
    # test_syn_data(epochs_list, data_manager.states, all_lm, all_gyro)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SSO overpass training data")
    parser.add_argument("--altitude", type=float, default=590e3,
                        help="Orbit altitude in metres (default 590000)")
    parser.add_argument("--ltan", type=float, default=11.0,
                        help="Local time of ascending node in hours (default 11.0)")
    parser.add_argument("--mgrs", type=str, default="17R",
                        help="MGRS GZD overpass filter (default 17R)")
    parser.add_argument("--nadir-att", dest="nadir_att", action="store_true", default=True,
                        help="Nadir-pointing initial attitude (default)")
    parser.add_argument("--random-att", dest="nadir_att", action="store_false",
                        help="Random initial attitude")
    parser.add_argument("--nadir-rate", dest="nadir_rate", action="store_true", default=True,
                        help="Nadir-pointing angular velocity (default)")
    parser.add_argument("--random-rate", dest="nadir_rate", action="store_false",
                        help="Random angular velocity")
    parser.add_argument("--rate-std", type=float, default=1.0,
                        help="Random rate std dev in dps (default 1.0)")
    parser.add_argument("--gyro-rate", type=float, default=1.0,
                        help="Gyro sampling rate in Hz (default 1.0)")
    parser.add_argument("--image-rate", type=float, default=1 / 15.0,
                        help="Image collection rate in Hz (default 1/30 = once per 30 seconds)")
    parser.add_argument("--duration", type=float, default=130.0,
                        help="Dataset duration in seconds (default 130.0)")
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of trials (default 1)")
    parser.add_argument("--ld-version", type=int, default=1,
                        help="Landmark detection model version (default 1, e.g. 1 for V1, 2 for V2)")
    parser.add_argument("--save-lat-lon", dest="save_lat_lon", action="store_true", default=False,
                        help="Save per-pixel lat/lon arrays as .npz alongside each image (default off)")
    parser.add_argument("--no-labels", dest="write_labels", action="store_false", default=True,
                        help="Skip YOLO label generation to speed up image creation")
    parser.add_argument("--cpu", dest="use_gpu", action="store_false", default=True,
                        help="Force CPU image simulation instead of CuPy/GPU")
    parser.add_argument("--gpu-preload-region", type=str, default=None,
                        help="Preload all GeoTIFF tiles for this region into RAM/VRAM (default: current mgrs)")
    args = parser.parse_args()

    load_brahe_data_files_if_needed()

    att_label = "nadir" if args.nadir_att else "random"

    for gzd, region_name in _MGRS_GZD_NAMES.items():
        run_ms = int(time.time() * 1000)
        run_dir = f"datasets/{gzd}_{region_name}_{att_label}_test"
        os.makedirs(run_dir, exist_ok=True)
        print(f"Run directory: {run_dir}")

        for _ in range(args.trials):
            run_simulation(
                run_dir=run_dir,
                run_ms=run_ms,
                altitude_m=args.altitude,
                ltan_h=args.ltan,
                mgrs_gzd=gzd,
                nadir_att=args.nadir_att,
                nadir_rate=args.nadir_rate,
                rate_std_dps=args.rate_std,
                gyro_rate_hz=args.gyro_rate,
                image_rate_hz=args.image_rate,
                duration_s=args.duration,
                ld_version=args.ld_version,
                save_lat_lon=args.save_lat_lon,
                write_labels=args.write_labels,
                use_gpu=args.use_gpu,
                gpu_preload_region=args.gpu_preload_region,
            )
