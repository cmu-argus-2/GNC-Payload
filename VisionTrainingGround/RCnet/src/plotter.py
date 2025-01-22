import matplotlib.pyplot as plt
import os

class Plotter:
    def __init__(self):
        self.loss_history = []

    def update_loss(self, loss):
        self.loss_history.append(loss)
        # self.plot_loss()

    def plot_loss(self):
        plt.clf()  # Clear the current figure
        plt.plot(self.loss_history, label='Training Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs. Epoch')
        plt.legend()
        plt.draw()
        plt.pause(0.001)

    def save_plot(self, file_path='loss_plot.png'):
        plt.clf()  # Clear the current figure
        plt.plot(self.loss_history, label='Training Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs. Epoch')
        plt.legend()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)
        print(f'Loss plot saved at {file_path}')
