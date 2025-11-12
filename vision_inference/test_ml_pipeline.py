import os

from image_simulation.test_earth_vis import simulate_image
from vision_inference.ml_pipeline import MLPipeline


def main():
    frame = simulate_image(altitude=510e3, display_image=False)
    pipeline = MLPipeline()
    landmark_detections, region_slices = pipeline.run_ml_pipeline_on_single(frame)
    save_dir = os.path.join(__file__, "../output")
    pipeline.visualize_landmarks(frame, landmark_detections, region_slices, save_dir=save_dir)


if __name__ == "__main__":
    main()
