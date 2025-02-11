"""
Region Classification Module

This module defines the RegionClassifier class, which leverages a pretrained EfficientNet model to classify
images based on geographic regions. The classifier is tailored to recognize specific regions by adjusting the
final layer to match the number of target classes and loading custom model weights. Main functionalities
include image preprocessing and the execution of classification, providing class probabilities for each
recognized region.


Author: Eddie
Date: [Creation or Last Update Date]
"""

import os
import cv2
from time import perf_counter
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from vision_inference.logger import Logger
from typing import List

from vision_inference.frame import Frame
from utils.config_utils import load_config


class RegionClassifier:
    NUM_CLASSES = 16
    CONFIDENCE_THRESHOLD = 0.55
    DOWNSAMPLED_SIZE = (224, 224)
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]
    MODEL_DIR = os.path.abspath(os.path.join(__file__, "../../models/rc"))
    MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, f"model_effnet_0.997_acc.pth")

    def __init__(self):
        Logger.log("INFO", "Initializing RegionClassifier.")

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = ClassifierEfficient().to(self.device)

            # Load Custom model weights
            self.model.load_state_dict(
                torch.load(RegionClassifier.MODEL_WEIGHTS_PATH, map_location=self.device)
            )
            self.model.eval()
            Logger.log("INFO", "Model loaded successfully.")

        except Exception as e:
            Logger.log("ERROR", f"Failed to load model: {e}")
            raise

        # Define the preprocessing
        self.transforms = transforms.Compose(
            [
                transforms.Resize(RegionClassifier.DOWNSAMPLED_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=RegionClassifier.IMAGE_NET_MEAN, std=RegionClassifier.IMAGE_NET_STD
                ),
            ]
        )

        self.region_ids = RegionClassifier.load_region_ids()

    @staticmethod
    def load_region_ids() -> List[str]:
        try:
            config = load_config()
            region_ids = config["vision"]["salient_mgrs_region_ids"]
            assert len(region_ids) == RegionClassifier.NUM_CLASSES, "Incorrect number of region IDs."
            assert len(set(region_ids)) == RegionClassifier.NUM_CLASSES, "Duplicate region IDs detected."
            return region_ids
        except Exception as e:
            Logger.log("ERROR", f"Configuration error: {e}")
            raise

    def classify_region(self, frame_obj: Frame) -> List[str]:
        Logger.log(
            "INFO",
            f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] Starting the classification process.",
        )
        try:
            img = Image.fromarray(cv2.cvtColor(frame_obj.frame, cv2.COLOR_BGR2RGB))
            img = self.transforms(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                start_time = perf_counter()
                outputs = self.model(img)
                inference_time = perf_counter() - start_time

                # TODO: are we accidentally applying sigmoid twice?
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > RegionClassifier.CONFIDENCE_THRESHOLD).float()
                predicted_indices = predicted.nonzero(as_tuple=True)[1]
                predicted_region_ids = [self.region_ids[idx] for idx in predicted_indices]

        except Exception as e:
            Logger.log("ERROR", f"Classification process failed: {e}")
            raise

        Logger.log(
            "INFO",
            f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] {predicted_region_ids} region(s) identified.",
        )
        Logger.log("INFO", f"Inference completed in {inference_time:.2f} seconds.")
        return predicted_region_ids


class ClassifierEfficient(nn.Module):
    def __init__(self):
        super(ClassifierEfficient, self).__init__()
        # Using new weights system
        # This uses the most up-to-date weights
        weights = EfficientNet_B0_Weights.DEFAULT
        self.efficientnet = efficientnet_b0(weights=weights)
        for param in self.efficientnet.features[:3].parameters():
            param.requires_grad = False
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_features, RegionClassifier.NUM_CLASSES)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.sigmoid(x)
        return x
