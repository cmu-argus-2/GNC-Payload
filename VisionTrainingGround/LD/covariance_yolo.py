"""
This script analyzes the error covariance in predicting pixel coordinates of landmarks
using a trained YOLO model. It compares predicted bounding box coordinates with ground truth
and calculates error statistics.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from utils.config_utils import load_config


def read_ground_truth(label_path, img_size):
    """
    Read ground truth bounding boxes from YOLO label file.

    Args:
        label_path (str): Path to YOLO label file
        img_size (tuple): Image dimensions (width, height)

    Returns:
        list: List of [class_id, cx, cy, w, h] in pixel coordinates
    """
    width, height = img_size
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f.readlines():
            values = line.strip().split()
            if len(values) == 5:
                class_id = int(values[0])
                # Convert normalized coordinates to pixel coordinates
                cx = float(values[1]) * width
                cy = float(values[2]) * height
                w = float(values[3]) * width
                h = float(values[4]) * height
                boxes.append([class_id, cx, cy, w, h])

    return boxes


def match_predictions_with_ground_truth(predictions, ground_truths, iou_threshold=0.5):
    """
    Match predictions with ground truth based on IoU and class.

    Args:
        predictions (list): List of [class_id, conf, cx, cy, w, h] for predictions
        ground_truths (list): List of [class_id, cx, cy, w, h] for ground truth
        iou_threshold (float): IoU threshold for matching

    Returns:
        list: Matched pairs of prediction and ground truth [pred_idx, gt_idx, iou]
    """
    matches = []

    # Convert to numpy arrays for easier calculations
    if not predictions or not ground_truths:
        return matches

    pred_boxes = np.array(
        [[p[2] - p[4] / 2, p[3] - p[5] / 2, p[2] + p[4] / 2, p[3] + p[5] / 2] for p in predictions]
    )
    gt_boxes = np.array(
        [
            [g[1] - g[3] / 2, g[2] - g[4] / 2, g[1] + g[3] / 2, g[2] + g[4] / 2]
            for g in ground_truths
        ]
    )

    for pred_idx, pred in enumerate(predictions):
        pred_class = pred[0]

        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            gt_class = gt[0]

            # Skip if classes don't match
            if pred_class != gt_class:
                continue

            # Calculate IoU
            px1, py1, px2, py2 = pred_boxes[pred_idx]
            gx1, gy1, gx2, gy2 = gt_boxes[gt_idx]

            x_left = max(px1, gx1)
            y_top = max(py1, gy1)
            x_right = min(px2, gx2)
            y_bottom = min(py2, gy2)

            if x_right < x_left or y_bottom < y_top:
                iou = 0
            else:
                intersection = (x_right - x_left) * (y_bottom - y_top)
                pred_area = (px2 - px1) * (py2 - py1)
                gt_area = (gx2 - gx1) * (gy2 - gy1)
                union = pred_area + gt_area - intersection
                iou = intersection / union if union > 0 else 0

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            matches.append([pred_idx, best_gt_idx, best_iou])

    return matches


def calculate_region_error_statistics(all_errors):
    """
    Calculate region-level error statistics across all classes.

    Args:
        all_errors (dict): Dictionary with class_ids as keys and lists of errors as values

    Returns:
        dict: Region-level error statistics
    """
    # Combine errors from all classes into a single list
    all_error_points = []
    for class_id, errors in all_errors.items():
        all_error_points.extend(errors)

    if not all_error_points:
        return None

    # Convert to numpy array
    errors_array = np.array(all_error_points)

    # Extract just the position errors (x, y) and ignore size errors
    position_errors = errors_array[:, :2]

    # Calculate position error statistics
    mean_position_error = np.mean(position_errors, axis=0)
    position_cov = np.cov(position_errors.T)

    # Calculate error magnitudes (Euclidean distance)
    error_magnitudes = np.sqrt(np.sum(position_errors**2, axis=1))
    mean_magnitude = np.mean(error_magnitudes)
    std_magnitude = np.std(error_magnitudes)

    # Calculate RMSE
    rmse_x = np.sqrt(np.mean(position_errors[:, 0] ** 2))
    rmse_y = np.sqrt(np.mean(position_errors[:, 1] ** 2))
    rmse_total = np.sqrt(rmse_x**2 + rmse_y**2)

    # Get 95% confidence interval
    sorted_magnitudes = np.sort(error_magnitudes)
    percentile_95 = np.percentile(error_magnitudes, 95)

    return {
        "count": len(all_error_points),
        "mean_position_error": mean_position_error,
        "position_cov": position_cov,
        "mean_magnitude": mean_magnitude,
        "std_magnitude": std_magnitude,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_total": rmse_total,
        "error_95_percentile": percentile_95,
    }


def analyze_landmark_errors(
    region, data_path, model_path, output_dir="error_analysis", conf_threshold=0.25
):
    """
    Main function to analyze landmark prediction errors.

    Args:
        region (str): Region code for the model
        data_path (str): Path to the dataset YAML file
        model_path (str): Path to the trained model file
        output_dir (str): Directory to save error analysis results
        conf_threshold (float): Confidence threshold for detections

    Returns:
        dict: Error statistics for each class
    """
    # Create region-specific output directory
    region_output_dir = os.path.join(output_dir, region)
    if not os.path.exists(region_output_dir):
        os.makedirs(region_output_dir)

    print(f"Analyzing region: {region}")

    # Load dataset configuration
    with open(data_path, "r") as f:
        dataset_config = yaml.safe_load(f)

    # Get validation image and label paths
    dataset_path = dataset_config.get("path", "")
    val_img_dir = os.path.join(dataset_path, dataset_config.get("val", "val/images"))
    val_label_dir = os.path.join(dataset_path, "val/labels")

    # Load trained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)

    # Prepare to collect errors
    all_errors = {}  # Dictionary to store errors for each class
    all_matches = []  # List to store all matches for analysis

    # Process validation images
    image_files = [f for f in os.listdir(val_img_dir) if f.endswith((".jpg", ".png"))]

    for img_file in tqdm(image_files, desc=f"Analyzing {region} images"):
        img_path = os.path.join(val_img_dir, img_file)
        label_path = os.path.join(val_label_dir, os.path.splitext(img_file)[0] + ".txt")

        # Skip if label file doesn't exist
        if not os.path.exists(label_path):
            continue

        # Get image dimensions
        img = Image.open(img_path)
        img_size = img.size  # (width, height)

        # Read ground truth
        ground_truths = read_ground_truth(label_path, img_size)

        # Run inference
        results = model(img_path, conf=conf_threshold)[0]

        # Extract predictions
        predictions = []
        for i, box in enumerate(results.boxes):
            cls = int(box.cls.item())
            conf = box.conf.item()

            # Extract box coordinates in pixel units
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            # Convert to center format [class, conf, cx, cy, width, height]
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]

            predictions.append([cls, conf, cx, cy, width, height])

        # Match predictions with ground truth
        matches = match_predictions_with_ground_truth(predictions, ground_truths)

        # Calculate errors for matched predictions
        for pred_idx, gt_idx, iou in matches:
            pred = predictions[pred_idx]
            gt = ground_truths[gt_idx]

            class_id = gt[0]

            # Calculate error in center coordinates and dimensions
            error = [
                pred[2] - gt[1],  # cx error
                pred[3] - gt[2],  # cy error
                pred[4] - gt[3],  # width error
                pred[5] - gt[4],  # height error
            ]

            # Initialize class in all_errors if not present
            if class_id not in all_errors:
                all_errors[class_id] = []

            all_errors[class_id].append(error)

    # Calculate region-level error statistics
    region_stats = calculate_region_error_statistics(all_errors)

    # Save region-level error statistics
    with open(os.path.join(region_output_dir, "region_error_statistics.txt"), "w") as f:
        f.write(f"Region: {region}\n\n")
        f.write(f"Total landmarks detected: {region_stats['count']}\n")
        f.write(
            f"Mean Position Error (x, y): [{region_stats['mean_position_error'][0]:.2f}, {region_stats['mean_position_error'][1]:.2f}] pixels\n"
        )
        f.write(
            f"Position Error Magnitude: {region_stats['mean_magnitude']:.2f} ± {region_stats['std_magnitude']:.2f} pixels\n"
        )
        f.write(f"RMSE-X: {region_stats['rmse_x']:.2f} pixels\n")
        f.write(f"RMSE-Y: {region_stats['rmse_y']:.2f} pixels\n")
        f.write(f"RMSE-Total: {region_stats['rmse_total']:.2f} pixels\n")
        f.write(f"95% of errors are below: {region_stats['error_95_percentile']:.2f} pixels\n\n")
        f.write(f"Position Error Covariance Matrix:\n")
        for row in region_stats["position_cov"]:
            f.write(f"  {row}\n")

    # Create a simple summary plot for the region
    plt.figure(figsize=(10, 8))
    errors = np.concatenate([np.array(errors)[:, :2] for errors in all_errors.values()])
    plt.scatter(errors[:, 0], errors[:, 1], alpha=0.5, color="blue")
    plt.grid(True)
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)
    plt.xlabel("Center X Error (pixels)")
    plt.ylabel("Center Y Error (pixels)")
    plt.title(f"Position Prediction Errors - Region {region}")

    # Draw error ellipse for 95% confidence
    from matplotlib.patches import Ellipse

    eigenvals, eigenvecs = np.linalg.eig(region_stats["position_cov"])
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width, height = 2 * np.sqrt(
        5.991 * eigenvals
    )  # 5.991 is chi-square value for 95% confidence with 2 DOF
    ellipse = Ellipse(
        xy=region_stats["mean_position_error"],
        width=width,
        height=height,
        angle=angle,
        edgecolor="red",
        fc="none",
        lw=2,
        label="95% Confidence",
    )
    plt.gca().add_patch(ellipse)
    plt.legend()
    plt.savefig(os.path.join(region_output_dir, "region_position_errors.png"))

    # Return region-level statistics
    return region_stats


def compare_regions(all_stats, output_dir):
    """
    Create comparative analysis across regions, focusing on region-level metrics only.
    Includes error covariance information in text format.

    Args:
        all_stats (dict): Dictionary with region codes as keys and region-level error stats as values
        output_dir (str): Directory to save comparison results
    """
    compare_dir = os.path.join(output_dir, "comparison")
    if not os.path.exists(compare_dir):
        os.makedirs(compare_dir)

    # Generate region comparison report
    with open(os.path.join(compare_dir, "region_comparison.txt"), "w") as f:
        f.write("Regional Comparison of Error Statistics\n")
        f.write("=" * 40 + "\n\n")

        # Sort regions by RMSE for ranking
        sorted_regions = sorted(all_stats.keys(), key=lambda r: all_stats[r]["rmse_total"])

        f.write("Region Rankings by Total RMSE:\n")
        f.write("-" * 30 + "\n")
        for i, region in enumerate(sorted_regions):
            stat = all_stats[region]
            f.write(
                f"{i+1}. Region {region}: RMSE = {stat['rmse_total']:.2f} pixels (from {stat['count']} landmarks)\n"
            )

        f.write("\n\n")

        # Calculate aggregate statistics
        total_landmarks = sum(stats["count"] for stats in all_stats.values())
        weighted_mean_x = (
            sum(stats["mean_position_error"][0] * stats["count"] for stats in all_stats.values())
            / total_landmarks
        )
        weighted_mean_y = (
            sum(stats["mean_position_error"][1] * stats["count"] for stats in all_stats.values())
            / total_landmarks
        )
        weighted_rmse = np.sqrt(
            sum(stats["rmse_total"] ** 2 * stats["count"] for stats in all_stats.values())
            / total_landmarks
        )

        # Calculate aggregate covariance matrix (weighted average)
        weighted_cov = np.zeros((2, 2))
        for region, stats in all_stats.items():
            weight = stats["count"] / total_landmarks
            weighted_cov += stats["position_cov"] * weight

        # Calculate eigenvalues and eigenvectors of the aggregate covariance
        eig_vals, eig_vecs = np.linalg.eig(weighted_cov)

        # Calculate principal axis angles and lengths
        angle = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))
        axis_lengths = 2 * np.sqrt(eig_vals)  # 2 standard deviations

        # Calculate correlation coefficient from the covariance matrix
        corr = (
            weighted_cov[0, 1] / np.sqrt(weighted_cov[0, 0] * weighted_cov[1, 1])
            if weighted_cov[0, 0] * weighted_cov[1, 1] > 0
            else 0
        )

        # Write aggregate statistics
        f.write("AGGREGATE STATISTICS ACROSS ALL REGIONS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Landmarks: {total_landmarks}\n")
        f.write(
            f"Weighted Mean Position Error: [{weighted_mean_x:.2f}, {weighted_mean_y:.2f}] pixels\n"
        )
        f.write(f"Weighted RMSE: {weighted_rmse:.2f} pixels\n\n")

        f.write("Error Covariance Analysis:\n")
        f.write(f"  Aggregate Error Covariance Matrix:\n")
        f.write(f"    [{weighted_cov[0, 0]:.4f}, {weighted_cov[0, 1]:.4f}]\n")
        f.write(f"    [{weighted_cov[1, 0]:.4f}, {weighted_cov[1, 1]:.4f}]\n\n")

        f.write(f"  X-Y Error Correlation: {corr:.4f}\n")
        f.write(f"  Principal Error Axes: {axis_lengths[0]:.2f} and {axis_lengths[1]:.2f} pixels\n")
        f.write(f"  Error Ellipse Orientation: {angle:.2f} degrees\n")
        f.write(
            f"  Error Anisotropy Ratio: {(axis_lengths[0]/axis_lengths[1] if axis_lengths[1] > 0 else 0):.2f} (major/minor axis)\n\n"
        )

        # Covariance interpretation
        f.write("Error Covariance Interpretation:\n")
        if abs(corr) < 0.2:
            f.write("  X and Y errors are largely uncorrelated\n")
        elif corr > 0:
            f.write("  Positive correlation: When X error increases, Y error tends to increase\n")
        else:
            f.write("  Negative correlation: When X error increases, Y error tends to decrease\n")

        if axis_lengths[0] / axis_lengths[1] > 2:
            f.write("  Strong directional bias: Errors are much larger along one axis\n")
        elif axis_lengths[0] / axis_lengths[1] > 1.2:
            f.write("  Moderate directional bias: Errors are somewhat larger along one axis\n")
        else:
            f.write("  No strong directional bias: Error distribution is nearly circular\n")

        f.write("\n")
        f.write("-" * 40 + "\n\n")

        # Write per-region statistics with covariance details for each region
        f.write("Detailed Statistics by Region:\n")
        f.write("-" * 30 + "\n\n")

        for region in sorted_regions:
            stat = all_stats[region]
            cov = stat["position_cov"]

            # Calculate region-specific metrics from covariance
            eig_vals_r, eig_vecs_r = np.linalg.eig(stat["position_cov"])
            angle_r = np.degrees(np.arctan2(eig_vecs_r[1, 0], eig_vecs_r[0, 0]))
            axis_lengths_r = 2 * np.sqrt(eig_vals_r)

            # Calculate correlation coefficient
            region_corr = (
                cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]) if cov[0, 0] * cov[1, 1] > 0 else 0
            )

            # Write region details
            f.write(f"Region {region}:\n")
            f.write(f"  Count: {stat['count']}\n")
            f.write(
                f"  Mean Position Error (x, y): [{stat['mean_position_error'][0]:.2f}, {stat['mean_position_error'][1]:.2f}] pixels\n"
            )
            f.write(f"  RMSE-Total: {stat['rmse_total']:.2f} pixels\n")
            f.write(f"  Position Error Covariance Matrix:\n")
            f.write(f"    [{cov[0, 0]:.4f}, {cov[0, 1]:.4f}]\n")
            f.write(f"    [{cov[1, 0]:.4f}, {cov[1, 1]:.4f}]\n")
            f.write(f"  X-Y Error Correlation: {region_corr:.4f}\n")
            f.write(
                f"  Principal Error Axes: {axis_lengths_r[0]:.2f} and {axis_lengths_r[1]:.2f} pixels\n"
            )
            f.write(f"  Error Ellipse Orientation: {angle_r:.2f} degrees\n")
            f.write(
                f"  Error Anisotropy Ratio: {(axis_lengths_r[0]/axis_lengths_r[1] if axis_lengths_r[1] > 0 else 0):.2f}\n\n"
            )


def main():
    """
    Main function to process multiple regions and generate results.
    """
    # Base paths
    base_dir = "/mnt/sdb2/training2/"  # "/home/argus/Arvind/GNC-Payload/VisionTrainingGround/LD"
    output_dir = base_dir + "/error_analysis_results"
    config = load_config()
    regions = config["vision"]["salient_mgrs_region_ids"]
    # regions = ["10S"]
    print(f"Loaded {len(regions)} regions from config")
    print(f"Regions to analyze: {', '.join(regions)}")
    # Collect stats for each region
    all_stats = {}
    for region in regions:
        # Construct paths for this region
        data_path = os.path.join(base_dir, region, "LD_training", "dataset.yaml")
        model_dir_pattern = os.path.join(base_dir, region, "yolo_training_results_" + region, "*")
        model_dirs = glob.glob(model_dir_pattern)
        if len(model_dirs) == 0:
            print(f"Skipping region {region}: No model directory found")
            continue
        model_dir = model_dirs[0]  # assume there is only one

        model_path = os.path.join(model_dir, "weights", "best.pt")

        # Skip if files don't exist
        if not os.path.exists(data_path):
            print(f"Skipping region {region}: Data files not found")
            continue
        if not os.path.exists(model_path):
            print(f"Skipping region {region}: Model file not found")
            continue

        # Run analysis
        stats = analyze_landmark_errors(
            region=region,
            data_path=data_path,
            model_path=model_path,
            output_dir=output_dir,
            conf_threshold=0.25,
        )

        all_stats[region] = stats

    # Create comparative analysis
    if len(all_stats) > 1:
        compare_regions(all_stats, output_dir)

    print(f"Analysis completed for {len(all_stats)} regions")


if __name__ == "__main__":
    main()
