import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from region_classifier import ImageClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate a region classifier.")
    
    # General flags
    parser.add_argument('--train_flag', action='store_true', help="Set this flag to enable training mode.")
    parser.add_argument('--save_plot_flag', action='store_true', help="Set this flag to save training plots.")
    
    # Paths
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument('--save_plot_path', type=str, default="plot.png", help="Path to save the training plot.")
    parser.add_argument('--model_save_path', type=str, default="model.pth", help="Path to save the trained model.")
    parser.add_argument('--model_load_path', type=str, default="model.pth", help="Path to load the pre-trained model.")
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate for the optimizer.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create the classifier object
    classifier = ImageClassifier(
        data_path=args.data_dir, 
        save_plot_flag=args.save_plot_flag, 
        save_plot_path=args.save_plot_path
    )
    
    if args.train_flag:
        # Train the model
        classifier.train(epochs=args.epochs, learning_rate=args.learning_rate)
        classifier.save_model(path=args.model_save_path)
    else:
        # Load the model for evaluation
        classifier.load_model(path=args.model_load_path)
        print("Model loaded successfully.")
    
    # Evaluate the model
    classifier.evaluate()
