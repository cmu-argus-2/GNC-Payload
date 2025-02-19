"""
Image Classification Module using EfficientNet.

This module defines the `ImageClassifier` class for training, 
evaluating, and validating an image classification model.  
It leverages EfficientNet-B0 as the backbone, supports logging with Weights & Biases (wandb),  
and provides utilities for dataset preparation, training, and performance evaluation.
"""

import os
from typing import List, Optional

import torch
import wandb
from data_loader import CustomImageDataset
from efficientnet_pytorch import EfficientNet
from plotter import Plotter
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from collections import defaultdict
import random

from sklearn.manifold import TSNE
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import numpy as np

# from add_dead_pixels import AddDeadPixels

from io import BytesIO

from PIL import Image

import time

class ImageClassifier:
    """
    A deep learning-based multi-label image classifier using EfficientNet.

    This class provides methods for preparing datasets, training, validation,
    evaluation, and saving/loading models. It integrates with Weights & Biases (wandb)
    for logging and visualization.

    Attributes:
        device (torch.device): The device (CPU or GPU) on which the model runs.
        train_loader (DataLoader): DataLoader for training dataset.
        test_loader (DataLoader): DataLoader for testing dataset.
        val_loader (DataLoader): DataLoader for validation dataset.
        model (torch.nn.Module): The deep learning model for classification.
        plotter (Plotter): Utility for plotting training loss.
        save_plot_flag (bool): Flag to determine whether to save training loss plots.
        save_plot_path (str): Path to save the loss plot.

    Methods:
        _prepare_data(data_path):
            Loads and preprocesses the dataset into DataLoaders.

        _initialize_model():
            Initializes the EfficientNet-B0 model with a modified classifier layer.

        train(epochs=10, learning_rate=1e-3):
            Trains the model and logs loss/accuracy using wandb.

        validate():
            Evaluates model performance on the validation dataset.

        evaluate(output_file="RCnet/results/evaluation_results.txt"):
            Evaluates the model on the test dataset and logs results.

        save_model(path="model.pth"):
            Saves the trained model's weights.

        load_model(path="model.pth"):
            Loads a saved model's weights and sets it to evaluation mode.
    """

    def __init__(
        self,
        data_path: str,
        selected_classes: Optional[List[str]] = None,
        save_plot_flag: bool = False,
        save_plot_path: Optional[str] = None,
    ) -> None:
        """
        Initializes the ImageClassifier.

        Args:
            data_path (str): Path to the dataset.
            selected_classes (list): List of selected classes for the multi-label classification.
            save_plot_flag (bool): Whether to save training loss plots.
            save_plot_path (str): Path to save the loss plot.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_data(data_path, selected_classes)
        self.model = EfficientNet.from_pretrained("efficientnet-b0")

        # Replace the classifier layer
        num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(num_ftrs, len(self.classes))  # Output for each class

        self.model = self.model.to(self.device)
        self.plotter = Plotter()
        self.save_plot_flag = save_plot_flag
        self.save_plot_path = save_plot_path

    def _prepare_data(self, data_path: str, selected_classes: Optional[List[str]]) -> None:
        """
        Prepares the dataset by applying transformations and loading it into DataLoaders.

        Args:
            data_path (str): Path to the dataset directory.
            selected_classes (list): List of salient regions for classification.
        """
        if selected_classes is None:
            # Use all available classes and output a warning
            selected_classes = sorted(os.listdir(data_path + "/train"))
            print("Warning: Using all available classes for training!")

        self.classes = selected_classes
        print("self.class", self.classes)

        # Define transforms for training and testing sets
        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                # transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.ToTensor(),
                # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
                
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                # transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.ToTensor(),
                # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
                
            ]
        )

        # Load datasets with appropriate transforms
        train_dataset = CustomImageDataset(
            root_dir=data_path + "/train", selected_classes=self.classes, transform=train_transform
        )
        test_dataset = CustomImageDataset(
            root_dir=data_path + "/test", selected_classes=self.classes, transform=test_transform
        )
        val_dataset = CustomImageDataset(
            root_dir=data_path + "/val", selected_classes=self.classes, transform=test_transform
        )

        # Create DataLoader objects for training and testing sets
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=True)
        print("Init Dataloaders")

    def train(self, epochs: int = 10, learning_rate: float = 1e-3) -> None:
        """
        Trains the image classifier using EfficientNet-B0 and logs progress using wandb.

        Args:
            epochs (int): Number of training epochs. Default is 10.
            learning_rate (float): Learning rate for optimization. Default is 1e-3.
        """
        wandb.init(
            project="RCnet",
            config={
                "mode": "train",
                "epochs": epochs,
                "learning_rate": learning_rate,
                "architecture": "EfficientNet-b0",
                "dataset": "Sentinel",
            },
        )

        criterion = nn.BCEWithLogitsLoss()  # BCE loss for multi-label classification
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0.0  # Initialize epoch loss
            # pylint: disable=unused-variable
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data = data.to(self.device)
                targets = targets.to(
                    self.device
                ).float()  # Ensure targets are float for BCEWithLogits
                scores = self.model(data)
                loss = criterion(scores, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                wandb.log({"batch_loss": loss.item()})
                epoch_loss += loss.item() * data.size(0)  # Accumulate batch loss

                # Log file names and class labels during training
                # with open('RCnet/results/training_results.txt', 'a') as f:
                #     for img_name, label in self.train_loader.dataset.files:
                #         f.write(f"{img_name}\t{label}\n")

            epoch_loss /= len(self.train_loader.dataset)  # Compute average batch loss
            print(f"Epoch [{epoch+1}/{epochs}], Avg. Loss: {epoch_loss:.4f}")
            wandb.log({"epoch": epoch, "loss": epoch_loss})
            self.plotter.update_loss(epoch_loss)

            if epoch % 2 == 0:
                self.save_model(path="RCnet/chkpts/model" + str(epoch + 1) + ".pth")
                self.validate()
            if epoch == epochs - 1:
                test_accuracy = self.evaluate()
                wandb.log({"test_accuracy": test_accuracy})

        if self.save_plot_flag:
            self.plotter.save_plot(self.save_plot_path)
            wandb.log({"loss_vs_epoch": wandb.Image(self.save_plot_path)})

    def save_model(self, path: str = "model.pth") -> None:
        """
        Saves the trained model to the specified path.

        Args:
            path (str): Path to save the model file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str = "model.pth") -> None:
        """
        Loads the trained model from the specified path.

        Args:
            path (str): Path to the saved model file.
        """
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()

    def validate(self) -> float:
        """
        Evaluates model performance on the validation dataset.

        Returns:
            float: Validation accuracy in percentage.
        """
        self.model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():  # No gradient is needed for validation
            for images, labels in self.val_loader:  # Use the validation data loader
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                predictions = torch.sigmoid(outputs) > 0.5  # Sigmoid + thresholding for multi-label
                total += labels.numel()
                correct += (predictions == labels).sum().item()

        accuracy = 100 * correct / total
        wandb.log({"validation_accuracy": accuracy})

        print(f"Validation Accuracy: {accuracy:.2f}%")
        return accuracy

    # pylint: disable=too-many-locals
    def evaluate(self, output_file: str = "RCnet/results/evaluation_results.txt") -> float:
        """
        Evaluates the model on the test dataset and logs results.

        Args:
            output_file (str): File path to save evaluation results.

        Returns:
            float: Test accuracy in percentage.
        """
        if wandb.run is None:
            wandb.init(
                project="RCnet",
                config={
                    "mode": "eval",
                    "architecture": "EfficientNet-b0",
                    "dataset": "Sentinel",
                },
            )

        self.model.eval()
        total_correct = 0
        total_labels = 0
        all_features = []
        all_labels = []
        class_correct = {i: 0 for i in range(40)}  # Assuming 40 classes
        class_total = {i: 0 for i in range(40)}

        class_images = {i: [] for i in range(40)}  # Store images per class
        tot_time=0
        with torch.no_grad():
            for batch in self.test_loader:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                start_time = time.time()
                outputs = self.model(images)
                end_time = time.time()
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()  # Multi-label thresholding
                tot_time+=end_time-start_time
                # Store features and labels for t-SNE
                all_features.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                # Group images by their predicted classes
                for i in range(images.size(0)):
                    predicted_classes = [j for j, val in enumerate(predicted[i]) if val == 1]
                    for pred_class in predicted_classes:
                        class_images[pred_class].append(images[i].cpu())

                # Compute accuracy
                true_positives = (predicted * labels).sum().item()
                total_correct += true_positives
                total_labels += labels.numel()

                # Compute per-class accuracy
                for class_idx in range(labels.size(1)):  # Iterate over classes
                    class_labels = labels[:, class_idx]
                    class_preds = predicted[:, class_idx]

                    class_correct[class_idx] += (class_labels * class_preds).sum().item()
                    class_total[class_idx] += class_labels.sum().item()

            # Convert collected features and labels to NumPy for t-SNE
            all_features = np.concatenate(all_features, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            # Log images per class
            for class_id, image_list in class_images.items():
                if len(image_list) > 0:
                    image_grid = torch.stack(image_list, dim=0)
                    class_name = self.classes[class_id]
                    wandb.log({
                        f"class_{class_name}_images": wandb.Image(image_grid,
                            caption=f"Class {class_name} predictions")
                    })

            # Compute class-wise accuracies
            class_accuracies = {
                self.classes[class_idx]: (100 * class_correct[class_idx] / class_total[class_idx]) if class_total[class_idx] > 0 else 0
                for class_idx in class_correct
            }

            # Plot bar chart for class-wise accuracies
            plt.figure(figsize=(12, 6))
            plt.bar(class_accuracies.keys(), class_accuracies.values(), color="blue")
            plt.xlabel("Class Index")
            plt.ylabel("Accuracy (%)")
            plt.title("Class-wise Accuracies")
            plt.xticks(ticks=range(len(self.classes)), labels=self.classes, rotation=90)  # Assuming 40 classes
            plt.ylim(0, 100)  # Accuracy range 0-100%

            # Save the figure and log to wandb
            plot_path = "RCnet/results/class_wise_accuracies.png"
            plt.savefig(plot_path)
            plt.close()
            wandb.log({"class_wise_accuracies_plot": wandb.Image(plot_path)})

            # Log overall accuracy and per-class accuracies
            wandb.log({"overall_accuracy": 100 * total_correct / total_labels, **{f"{k}_accuracy": v for k, v in class_accuracies.items()}})

            # Plot t-SNE visualization
            self.plot_tsne(all_features, all_labels, num_classes=40)

        accuracy = 100 * total_correct / total_labels
        print(f"Accuracy of the network on the test images: {accuracy:.2f}%")
        print(f"Total Inf time:{tot_time}")

        return accuracy

        
    
    def plot_tsne(self, features, labels, num_classes):
        """
        Generates and logs a t-SNE plot for the given features and labels to wandb.

        Args:
            features (ndarray): Feature representations of the dataset.
            labels (ndarray): Multi-label ground truth labels.
            num_classes (int): Number of classes in the dataset.

        Returns:
            None
        """
        # Reduce to 2D using t-SNE
        tsne_plot = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_results = tsne_plot.fit_transform(features)

        # Assign colors for multi-labels
        base_colors = [hsv_to_rgb([i / num_classes, 1, 1]) for i in range(num_classes)]

        def get_mixed_color(label_vector):
            """Mix colors based on label activation."""
            active_classes = np.where(label_vector == 1)[0]
            if len(active_classes) == 1:
                return base_colors[active_classes[0]]
            elif len(active_classes) > 1:
                return np.mean([base_colors[i] for i in active_classes], axis=0)
            return [0, 0, 0]  # Default color for no class

        colors = np.array([get_mixed_color(label) for label in labels])

        # Plot t-SNE
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, alpha=0.7, edgecolors="k")
        plt.title("t-SNE Visualization of Model Features")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.grid(True)

        # Save plot to memory using BytesIO
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)  # Rewind buffer for reading

        # Open the image from the buffer with PIL
        img = Image.open(buf)

        # Log the image to wandb
        wandb.log({"t-SNE Plot": wandb.Image(img)})

        buf.close()  # Close the buffer

        # Close the plot to avoid memory issues
        plt.close()