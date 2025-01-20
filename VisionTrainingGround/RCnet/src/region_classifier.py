import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from data_loader import CustomImageDataset
from plotter import Plotter
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import wandb

class ImageClassifier:
    def __init__(self, train_path, test_path, save_plot_flag, save_plot_path,val_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(test_path)
        self._prepare_data(train_path, test_path,val_path)
        self._initialize_model()
        self.plotter = Plotter() 
        self.save_plot_flag = save_plot_flag
        self.save_plot_path = save_plot_path


    # def print_first_layer_weights(self):
    #     # Access the model directly with self.model
    #     first_layer_weights = next(self.model.parameters()).detach().cpu().numpy()
    #     print(first_layer_weights.flatten()[0:5])  # Print the first few weights



    def _prepare_data(self, train_path, test_path, val_path):
        # Define transforms for training and testing sets
        train_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    # transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
                ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Load datasets with appropriate transforms
        train_dataset = CustomImageDataset(root_dir=train_path, transform=train_transform)
        test_dataset = CustomImageDataset(root_dir=test_path, transform=test_transform)
        val_dataset = CustomImageDataset(root_dir=val_path, transform=test_transform)

        # Create DataLoader objects for training and testing sets
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True)


    # def _initialize_model(self):
    #     self.model = models.resnet18(pretrained=True)
    #     print(len(self.train_loader.dataset.classes))
    #     self.model.fc = nn.Linear(self.model.fc.in_features, len(self.train_loader.dataset.classes))
    #     self.model = self.model.to(self.device)
    def _initialize_model(self):
        # Load EfficientNet-B0
        self.model = EfficientNet.from_pretrained('efficientnet-b0')

        # Replace the classifier layer
        num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(num_ftrs, len(self.train_loader.dataset.classes))

        self.model = self.model.to(self.device)
        # print("Weights before loading custom model:")
        # self.print_first_layer_weights()

    def train(self, epochs=10, learning_rate=1e-3):
        wandb.init(project="RCnet", config={
            "epochs": epochs,
            "learning_rate": learning_rate,
            "architecture": "EfficientNet-b0",
            "dataset": "Sentinel",
        })

        # Save the model's config
        config = wandb.config

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0.0  # Initialize epoch loss
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
                with open('training_results.txt', 'a') as f:
                    for img_name, label in self.train_loader.dataset.files:
                        f.write(f"{img_name}\t{label}\n")

            epoch_loss /= len(self.train_loader.dataset)  # Compute average batch loss
            print(f'Epoch [{epoch+1}/{epochs}], Avg. Loss: {epoch_loss:.4f}')
            wandb.log({"epoch": epoch, "loss": epoch_loss})

            if epoch % 10 == 0:
                self.validate()
            if epoch % epochs ==0: 
                test_accuracy = self.evaluate()
                wandb.log({"test_accuracy": test_accuracy})

        if self.save_plot_flag:
            self.plotter.save_plot(self.save_plot_path)
            wandb.log({"loss_vs_epoch": wandb.Image(self.save_plot_path)})


    def save_model(self, path='model.pth'):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='model.pth'):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        # print("Weights after loading custom model:")
        # self.print_first_layer_weights()

    # def evaluate(self):
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for data in self.test_loader:
    #             images, labels = data
    #             images, labels = images.to(self.device), labels.to(self.device)
    #             outputs = self.model(images)
    #             print(outputs)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
    def validate(self):
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

        print(f'Validation Accuracy: {accuracy:.2f}%')
        return accuracy

    def evaluate(self, output_file='evaluation_results.txt'):
        self.model.eval()  # Ensure the model is in evaluation mode
        correct = 0
        total = 0
        processed_images = 0  # Initialize a counter for processed images

        with open(output_file, 'w') as f:
            f.write("Image Name\tActual Class\tPredicted Class\tProbabilities\n")

            with torch.no_grad():
                for batch in self.test_loader:
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    print(labels)

                    outputs = self.model(images)
                    probabilities = F.softmax(outputs, dim=1)  # Apply softmax to convert to probabilities
                    _, predicted = torch.max(probabilities, 1)  # Get the class with the highest probability

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # Correctly calculate the index for image names
                    for i in range(images.size(0)):
                        global_index = processed_images + i
                        image_name = os.path.basename(self.test_loader.dataset.files[global_index][0])
                        actual_class = self.test_loader.dataset.classes[labels[i]]
                        predicted_class = self.test_loader.dataset.classes[predicted[i]]
                        probs = ", ".join([f"{p:.4f}" for p in probabilities[i]])

                        f.write(f"{image_name}\t{actual_class}\t{predicted_class}\t{probs}\n")

                    processed_images += images.size(0)  # Update the counter after each batch

                    

        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the test images: {accuracy:.2f}%')

        return accuracy



    