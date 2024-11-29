from jsonargparse import CLI

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from rich.progress import track
import matplotlib.pyplot as plt
from time import sleep


# External cost function to compute the loss
def compute_loss(pred1, pred2, target1, target2):
    # L2 loss (mean squared error)
    loss1 = torch.nn.functional.mse_loss(pred1, target1)
    loss2 = torch.nn.functional.mse_loss(pred2, target2)
    return loss1 + loss2


# Model class definition
class ImageReconstructionModel(nn.Module):
    def __init__(self, width, height, shift: int = 5):
        super().__init__()
        # Define two learnable matrices A and B
        self.shift = shift
        self.A = nn.Parameter(torch.zeros(1, 1, height, width+self.shift))  # Matrix A, 256x256 grayscale image
        self.B = nn.Parameter(torch.zeros(1, 1, height, width+self.shift))  # Matrix B, 256x256 grayscale image

    def forward(self):
        # Predict the first and second images based on A and B
        a = torch.sigmoid(self.A)
        b = torch.sigmoid(self.B)
        a1 = a[:, :, :, 0:-self.shift]
        a2 = a[:, :, :, self.shift:]
        b1 = b[:, :, :, 0:-self.shift]
        b2 = b[:, :, :, self.shift:]
        pred1 = torch.multiply(a1, b2)
        pred2 = torch.multiply(a2, b1)
        return pred1, pred2

    def get_filters(self):
        a = torch.sigmoid(self.A).detach().numpy()
        b = torch.sigmoid(self.B).detach().numpy()
        return a, b


# Function to load and preprocess images (convert to grayscale and resize)
def load_image(image_path, width, height):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((height, width)),  # Resize to 256x256 for consistency
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


# Main training loop
def train_model(
        target1, target2,
        width: int = 512,
        height: int = 512,
        shift: int = 5,
        num_epochs: int = 500,
        learning_rate: float = 1e-3,
        silent: bool = False
):
    # Initialize model and optimizer
    model = ImageReconstructionModel(width=width, height=height, shift=shift)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in track(range(num_epochs), description='training', disable=silent):
        # Zero gradients
        pred1, pred2 = model()
        loss = compute_loss(pred1, pred2, target1, target2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Return the learned matrices A and B
    return *model.get_filters(), pred1.detach().numpy(), pred2.detach().numpy(),


# Visualize the results
def visualize_results(A, B, pred1, pred2):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Show the matrices A and B
    ax[0, 0].imshow(A[0, 0], cmap='gray')
    ax[0, 0].set_title('Matrix A')

    ax[0, 1].imshow(B[0, 0], cmap='gray')
    ax[0, 1].set_title('Matrix B')

    # Show the predicted images
    ax[1, 0].imshow(pred1[0, 0], cmap='gray')
    ax[1, 0].set_title('Predicted Image 1')

    ax[1, 1].imshow(pred2[0, 0], cmap='gray')
    ax[1, 1].set_title('Predicted Image 2')

    plt.show()


def evaluate(
        target_image1_path: str,
        target_image2_path: str,
        filter_image1_path: str,
        filter_image2_path: str,
        width: int = 512,
        height: int = 512,
        shift: int = 10,
        iterations: int = 10000,
        silent: bool = False
):
    # Example usage
    target1 = load_image(target_image1_path, width=width, height=height)
    target2 = load_image(target_image2_path, width=width, height=height)
    A, B, pred1, pred2 = train_model(
        target1, target2,
        width=width, height=height, shift=shift,
        num_epochs=iterations,
        silent=silent
    )
    filter_a = Image.fromarray(np.uint8(A[0, 0] * 255))
    filter_b = Image.fromarray(np.uint8(B[0, 0] * 255))
    filter_a.save(filter_image1_path)
    filter_b.save(filter_image2_path)
    # Visualize the learned matrices and predicted images
    visualize_results(A, B, pred1, pred2)


def main():
    CLI(evaluate)


if __name__ == '__main__':
    main()
