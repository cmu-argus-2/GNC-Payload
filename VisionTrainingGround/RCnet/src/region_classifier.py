"""
Image Classification Module using EfficientNet.

This module defines the `ImageClassifier` class for training, 
evaluating, and validating an image classification model.  
It leverages EfficientNet-B0 as the backbone, supports logging with Weights & Biases (wandb),  
and provides utilities for dataset preparation, training, and performance evaluation.
"""

import os
import wandb
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from data_loader import CustomImageDataset
from plotter import Plotter


# pylint: disable=too-many-instance-attributes
class ImageClassifier:
    """
    A deep learning-based image classifier using EfficientNet.

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

    def __init__(self, data_path, save_plot_flag, save_plot_path):
        """
        Initializes the ImageClassifier.

        Args:
            data_path (str): Path to the dataset.
            save_plot_flag (bool): Whether to save training loss plots.
            save_plot_path (str): Path to save the loss plot.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_data(data_path)
        # Load EfficientNet-B0
        self.model = EfficientNet.from_pretrained("efficientnet-b0")

        # Replace the classifier layer
        num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(num_ftrs, len(self.train_loader.dataset.classes))

        self.model = self.model.to(self.device)
        self.plotter = Plotter()
        self.save_plot_flag = save_plot_flag
        self.save_plot_path = save_plot_path

    def _prepare_data(self, data_path):
        """
        Prepares the dataset by applying transformations and loading it into DataLoaders.

        Args:
            data_path (str): Path to the dataset directory.
        """
        # Define transforms for training and testing sets
        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                # transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        # Load datasets with appropriate transforms
        train_dataset = CustomImageDataset(root_dir=data_path + "/train", transform=train_transform)
        test_dataset = CustomImageDataset(root_dir=data_path + "/test", transform=test_transform)
        val_dataset = CustomImageDataset(root_dir=data_path + "/val", transform=test_transform)

        # Create DataLoader objects for training and testing sets
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=True)
        print("Init Dataloaders")

    def train(self, epochs=10, learning_rate=1e-3):
        """
        Trains the image classifier using EfficientNet-B0 and logs progress using wandb.

        Args:
            epochs (int): Number of training epochs. Default is 10.
            learning_rate (float): Learning rate for optimization. Default is 1e-3.
        """
        wandb.init(
            project="RCnet",
            config={
                "epochs": epochs,
                "learning_rate": learning_rate,
                "architecture": "EfficientNet-b0",
                "dataset": "Sentinel",
            },
        )

        # Save the model's config
        # config = wandb.config

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0.0  # Initialize epoch loss
            # pylint: disable=unused-variable
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)
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

            if epoch % 10 == 0:
                self.validate()
            if epoch == epochs - 1:
                test_accuracy = self.evaluate()
                wandb.log({"test_accuracy": test_accuracy})

        if self.save_plot_flag:
            self.plotter.save_plot(self.save_plot_path)
            wandb.log({"loss_vs_epoch": wandb.Image(self.save_plot_path)})

    def save_model(self, path="model.pth"):
        """
        Saves the trained model to the specified path.

        Args:
            path (str): Path to save the model file.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="model.pth"):
        """
        Loads the trained model from the specified path.

        Args:
            path (str): Path to the saved model file.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def validate(self):
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
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        wandb.log({"validation_accuracy": accuracy})

        print(f"Validation Accuracy: {accuracy:.2f}%")
        return accuracy

    # pylint: disable=too-many-locals
    def evaluate(self, output_file="RCnet/results/evaluation_results.txt"):
        """
        Evaluates the model on the test dataset and logs results.

        Args:
            output_file (str): File path to save evaluation results.

        Returns:
            float: Test accuracy in percentage.
        """
        self.model.eval()  # Ensure the model is in evaluation mode
        correct = 0
        total = 0
        processed_images = 0  # Initialize a counter for processed images

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Image Name\tActual Class\tPredicted Class\tProbabilities\n")

            with torch.no_grad():
                for batch in self.test_loader:
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    print(labels)

                    outputs = self.model(images)
                    probabilities = F.softmax(
                        outputs, dim=1
                    )  # Apply softmax to convert to probabilities
                    _, predicted = torch.max(
                        probabilities, 1
                    )  # Get the class with the highest probability

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # Correctly calculate the index for image names
                    for i in range(images.size(0)):
                        global_index = processed_images + i
                        image_name = os.path.basename(
                            self.test_loader.dataset.files[global_index][0]
                        )
                        actual_class = self.test_loader.dataset.classes[labels[i]]
                        # pylint: disable=unsubscriptable-object
                        predicted_class = self.test_loader.dataset.classes[predicted[i]]
                        probs = ", ".join([f"{p:.4f}" for p in probabilities[i]])

                        f.write(f"{image_name}\t{actual_class}\t{predicted_class}\t{probs}\n")

                    processed_images += images.size(0)  # Update the counter after each batch

        accuracy = 100 * correct / total
        print(f"Accuracy of the network on the test images: {accuracy:.2f}%")

        return accuracy
