"""
Unified pipeline: run test_batch_opt_data_gen simulation, then assemble one
MP4 video per camera (xp, yp, xm, ym) from the generated frames.

The relationship between args is simple:
    - you specify video duration, fps, and image_rate
    - frames_needed   = video_duration_s * fps         (e.g. 30 * 5 = 150)
    - sim_duration_s  = frames_needed / image_rate_hz  (e.g. 150 / 5 = 30s of orbit)
    - image_rate controls temporal resolution: higher = smoother but longer sim

Usage example:
    CUDA_VISIBLE_DEVICES=0 python create_video_from_frames.py \
        --output_dir datasets \
        --mgrs 17R \
        --fps 5 \
        --video-duration 30 \
        --image-rate 5.0 \
        --gpu-preload-region 17R
    CUDA_VISIBLE_DEVICES=0 python -m orbit_determination.create_video_from_frames \
    --mgrs 17R --fps 5 --video-duration 30 --image-rate 2.0 
"""

import argparse
import math
import os
import shutil
import time

import cv2

DEFAULT_IMAGE_RATE_HZ = 5.0

# Must match _CAM_NAME_TO_STR in test_batch_opt_data_gen.py
# Must match _CAM_NAME_TO_ID in test_batch_opt_data_gen.py: x+->0, y+->1, y-->2, x-->3
CAMERA_IDS = ["0", "1", "2", "3"]
CAMERA_LABELS = {"0": "xp", "1": "yp", "2": "ym", "3": "xm"}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def run_data_gen(
    run_dir: str,
    run_ms: int,
    mgrs_gzd: str,
    altitude_m: float,
    ltan_h: float,
    nadir_att: bool,
    nadir_rate: bool,
    rate_std_dps: float,
    gyro_rate_hz: float,
    image_rate_hz: float,
    video_duration_s: float,
    fps: int,
    ld_version: int,
    save_lat_lon: bool,
    write_labels: bool,
    use_gpu: bool,
    gpu_preload_region: str | None,
    trials: int,
) -> None:
    """
    Import and call run_simulation from test_batch_opt_data_gen directly.

    The simulation runs for exactly duration_s seconds at image_rate_hz,
    producing duration_s * image_rate_hz frames per camera.
    """
    import sys
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from utils.brahe_utils import load_brahe_data_files_if_needed
    from orbit_determination.test_batch_opt_data_gen import run_simulation

    load_brahe_data_files_if_needed()

    frames_needed = math.ceil(video_duration_s * fps)
    sim_duration_s = frames_needed / image_rate_hz

    os.makedirs(run_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"[DataGen] Starting simulation")
    print(f"  Run dir          : {run_dir}")
    print(f"  MGRS GZD         : {mgrs_gzd}")
    print(f"  Altitude         : {altitude_m/1e3:.1f} km")
    print(f"  LTAN             : {ltan_h:.1f} h")
    print(f"  Image rate       : {image_rate_hz:.4f} Hz (1 frame / {1/image_rate_hz:.2f} s)")
    print(f"  Frames needed    : {frames_needed}")
    print(f"  Sim duration     : {sim_duration_s:.1f} s  (= frames_needed / image_rate_hz)")
    print(f"  Cameras          : {list(CAMERA_IDS)}")
    print(f"  Trials           : {trials}")
    print(f"{'='*60}\n")

    for trial in range(trials):
        print(f"[DataGen] Trial {trial + 1}/{trials}")
        run_simulation(
            run_dir=run_dir,
            run_ms=run_ms,
            altitude_m=altitude_m,
            ltan_h=ltan_h,
            mgrs_gzd=mgrs_gzd,
            nadir_att=nadir_att,
            nadir_rate=nadir_rate,
            rate_std_dps=rate_std_dps,
            gyro_rate_hz=gyro_rate_hz,
            image_rate_hz=image_rate_hz,
            duration_s=sim_duration_s,
            ld_version=ld_version,
            save_lat_lon=save_lat_lon,
            write_labels=write_labels,
            use_gpu=use_gpu,
            gpu_preload_region=gpu_preload_region,
        )


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------

def create_videos_from_frames(
    run_dir: str,
    output_dir: str,
    region: str,
    fps: int,
    image_rate_hz: float = DEFAULT_IMAGE_RATE_HZ,
    gpu_preload_region: str | None = None,
    use_gpu: bool = True,
) -> None:
    """
    Assemble one MP4 per camera from JPG frames produced by run_simulation.

    Frames in run_dir are named:  raw_{timestamp_ms}_{cam_id}.jpg
    where cam_id is one of:       0 (xp), 1 (yp), 2 (ym), 3 (xm)

    Each frame captured at image_rate_hz is played back at fps.
    Video duration = total_frames / fps = (sim_duration * image_rate_hz) / fps
    """
    if use_gpu and gpu_preload_region is not None:
        _gpu_preload(gpu_preload_region)

    all_jpgs = sorted([
        f for f in os.listdir(run_dir)
        if f.lower().endswith(".jpg")
    ])

    if not all_jpgs:
        raise ValueError(f"No JPG files found in {run_dir}")

    # Bucket by cam_id: raw_{timestamp_ms}_{cam_id}.jpg
    cam_files: dict[str, list[str]] = {cam: [] for cam in CAMERA_IDS}
    for fname in all_jpgs:
        for cam in CAMERA_IDS:
            if fname.endswith(f"_{cam}.jpg"):
                cam_files[cam].append(os.path.join(run_dir, fname))
                break

    for cam in CAMERA_IDS:
        if not cam_files[cam]:
            print(f"[Video] WARNING: no frames found for camera {cam}, skipping.")
            continue
        label = CAMERA_LABELS[cam]
        _assemble_camera_video(
            cam_str=label,
            jpg_files=cam_files[cam],
            output_dir=output_dir,
            region=region,
            fps=fps,
            image_rate_hz=image_rate_hz,
            use_gpu=use_gpu,
        )


def _assemble_camera_video(
    cam_str: str,
    jpg_files: list[str],
    output_dir: str,
    region: str,
    fps: int,
    image_rate_hz: float,
    use_gpu: bool,
) -> None:
    """
    Copy frames into a staging directory with sequential names then call
    ffmpeg to encode H.264 MP4 with yuv420p — universally playable.
    """
    import subprocess

    total_frames = len(jpg_files)
    video_duration_s = total_frames / fps

    print(f"\n{'='*60}")
    print(f"[Video:{cam_str}]")
    print(f"  Source frames   : {total_frames}")
    print(f"  Playback FPS    : {fps}")
    print(f"  Video duration  : {video_duration_s:.2f} s")
    print(f"{'='*60}\n")

    # Staging directory with sequentially named frames for ffmpeg
    video_frames_dir = os.path.join(output_dir, f"{region}_{cam_str}_frames")
    if os.path.exists(video_frames_dir):
        shutil.rmtree(video_frames_dir)
    os.makedirs(video_frames_dir, exist_ok=True)

    print(f"[Video:{cam_str}] Copying {total_frames} frames to staging dir...")
    for i, src_path in enumerate(jpg_files):
        shutil.copy2(src_path, os.path.join(video_frames_dir, f"frame_{i:06d}.jpg"))
        if (i + 1) % max(1, total_frames // 10) == 0:
            print(f"  {100*(i+1)/total_frames:.0f}%  ({i+1}/{total_frames})")

    video_path = os.path.join(output_dir, f"{region}_{cam_str}_{fps}fps.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(video_frames_dir, "frame_%06d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",          # quality: 0=lossless, 23=default, 51=worst
        "-preset", "fast",
        video_path,
    ]

    print(f"[Video:{cam_str}] Running ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg stderr:\n{result.stderr}")
        raise RuntimeError(f"ffmpeg failed for camera {cam_str}")

    print(f"\n[Video:{cam_str}] Done!")
    print(f"  Output     : {video_path}")
    print(f"  Duration   : {video_duration_s:.2f} s")
    print(f"  Frames     : {total_frames}")
    print(f"  FPS        : {fps}")
    print(f"  Frame dir  : {video_frames_dir}")


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _gpu_preload(region: str) -> None:
    try:
        import cupy as cp
        print(f"[GPU] Preloading CuPy context for region {region}...")
        _ = cp.zeros((1,), dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
        print("[GPU] CuPy context ready.")
    except ImportError:
        print("[GPU] CuPy not available — skipping GPU preload.")


# _read_frame and _make_writer removed — ffmpeg handles encoding directly


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_MGRS_GZD_NAMES = {
    "10S": "California",
    "10T": "Oregon",
    "11R": "Baja",
    "12R": "Sonora",
    "16T": "Minnesota",
    "17R": "Florida",
    "17T": "Toronto",
    "18S": "NewJersey",
    "32S": "Tunisia",
    "32T": "Switzerland",
    "33S": "Sicilia",
    "33T": "Italy",
    "52S": "Korea",
    "53S": "Hiroshima",
    "54S": "Tokyo",
    "54T": "Sapporo",
}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run test_batch_opt_data_gen simulation then assemble one MP4 per "
            "camera (xp, yp, xm, ym). "
            "Simulation runs for --duration seconds at --image-rate Hz, "
            "producing duration * image_rate unique frames per camera, "
            "played back at --fps."
        )
    )

    # ---- Output / region ----
    parser.add_argument("--output_dir", type=str, default="datasets",
                        help="Base output directory (default: datasets).")
    parser.add_argument("--mgrs", type=str, default="17R",
                        help="MGRS GZD (default: 17R).")

    # ---- Video + simulation (unified by duration) ----
    parser.add_argument("--fps", type=int, required=True,
                        help="Playback frame rate of the output videos.")
    parser.add_argument("--video-duration", type=float, required=True,
                        dest="video_duration",
                        help="Desired video duration in seconds. "
                             "Sim duration is derived as ceil(video_duration * fps) / image_rate_hz.")
    parser.add_argument("--image-rate", type=float, default=DEFAULT_IMAGE_RATE_HZ,
                        dest="image_rate_hz",
                        help=f"Image capture rate in Hz (default {DEFAULT_IMAGE_RATE_HZ} Hz). "
                             f"Set equal to --fps for a 1:1 simulation-to-video mapping.")

    # ---- Data gen ----
    parser.add_argument("--altitude", type=float, default=590e3)
    parser.add_argument("--ltan", type=float, default=11.0)
    parser.add_argument("--nadir-att", dest="nadir_att", action="store_true", default=True)
    parser.add_argument("--random-att", dest="nadir_att", action="store_false")
    parser.add_argument("--nadir-rate", dest="nadir_rate", action="store_true", default=True)
    parser.add_argument("--random-rate", dest="nadir_rate", action="store_false")
    parser.add_argument("--rate-std", type=float, default=1.0)
    parser.add_argument("--gyro-rate", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--ld-version", type=int, default=1, dest="ld_version")
    parser.add_argument("--save-lat-lon", dest="save_lat_lon", action="store_true", default=False)
    parser.add_argument("--no-labels", dest="write_labels", action="store_false", default=True)

    # ---- GPU ----
    parser.add_argument("--gpu-preload-region", type=str, default=None,
                        dest="gpu_preload_region",
                        help="MGRS GZD to preload into RAM/VRAM (default: same as --mgrs).")
    parser.add_argument("--cpu", dest="use_gpu", action="store_false", default=True)

    # ---- Pipeline control ----
    parser.add_argument("--video-only", dest="video_only", action="store_true", default=False,
                        help="Skip data gen, assemble videos from existing frames.")
    parser.add_argument("--data-gen-only", dest="data_gen_only", action="store_true", default=False,
                        help="Run data gen only, skip video assembly.")

    args = parser.parse_args()

    region_name = _MGRS_GZD_NAMES.get(args.mgrs, args.mgrs)
    att_label = "nadir" if args.nadir_att else "random"
    run_dir = os.path.join(args.output_dir, f"{args.mgrs}_{region_name}_{att_label}_test")
    run_ms = int(time.time() * 1000)
    gpu_preload_region = args.gpu_preload_region or args.mgrs

    frames_needed = math.ceil(args.video_duration * args.fps)
    sim_duration_s = frames_needed / args.image_rate_hz
    print(f"\n[Pipeline] Summary")
    print(f"  Video duration  : {args.video_duration:.1f} s @ {args.fps} fps  →  {frames_needed} frames needed")
    print(f"  Image rate      : {args.image_rate_hz} Hz  →  sim duration = {sim_duration_s:.1f} s")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Data generation
    # ------------------------------------------------------------------
    if not args.video_only:
        run_data_gen(
            run_dir=run_dir,
            run_ms=run_ms,
            mgrs_gzd=args.mgrs,
            altitude_m=args.altitude,
            ltan_h=args.ltan,
            nadir_att=args.nadir_att,
            nadir_rate=args.nadir_rate,
            rate_std_dps=args.rate_std,
            gyro_rate_hz=args.gyro_rate,
            image_rate_hz=args.image_rate_hz,
            video_duration_s=args.video_duration,
            fps=args.fps,
            ld_version=args.ld_version,
            save_lat_lon=args.save_lat_lon,
            write_labels=args.write_labels,
            use_gpu=args.use_gpu,
            gpu_preload_region=gpu_preload_region,
            trials=args.trials,
        )

    # ------------------------------------------------------------------
    # Step 2: Video assembly (one per camera)
    # ------------------------------------------------------------------
    if not args.data_gen_only:
        create_videos_from_frames(
            run_dir=run_dir,
            output_dir=args.output_dir,
            region=region_name,
            fps=args.fps,
            image_rate_hz=args.image_rate_hz,
            gpu_preload_region=gpu_preload_region,
            use_gpu=args.use_gpu,
        )


if __name__ == "__main__":
    main()