from jsonargparse import CLI

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from rich.progress import track
import matplotlib.pyplot as plt
from typing import Optional, Tuple


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
        self.A = nn.Parameter(torch.ones(1, 1, height, width+self.shift) * -1)  # Matrix A, 256x256 grayscale image
        self.B = nn.Parameter(torch.ones(1, 1, height, width+self.shift) * -1)  # Matrix B, 256x256 grayscale image

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


class SRGBToLinearTransform:
    def __call__(self, img):
        # Apply the power function approximation for linearization
        # img = (img + 0.055) / 1.055
        img = img ** 2.4
        return img


# Function to load and preprocess images (convert to grayscale and resize)
def load_image(image_path, width, height):
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    transform = transforms.Compose([
        transforms.Resize((height, width)),  # Resize to 256x256 for consistency
        transforms.ToTensor(),
        SRGBToLinearTransform(),
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
) -> Tuple:
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
    return (torch.sigmoid(model.A).detach().numpy(),
            torch.sigmoid(model.B).detach().numpy(),
            pred1.detach().numpy(),
            pred2.detach().numpy(),)


# Visualize the results
def visualize_results(A, B, pred1, pred2):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Show the matrices A and B
    ax[0, 0].imshow(A[0, 0], cmap='gray')
    ax[0, 0].set_title('Matrix A')

    ax[0, 1].imshow(B[0, 0], cmap='gray')
    ax[0, 1].set_title('Matrix B')

    # Show the predicted images
    ax[1, 0].imshow(np.pow(pred1[0, 0], 1./2.4), cmap='gray')
    ax[1, 0].set_title('Predicted Image 1')

    ax[1, 1].imshow(np.pow(pred2[0, 0], 1./2.4), cmap='gray')
    ax[1, 1].set_title('Predicted Image 2')

    plt.show()


def evaluate(
        target1: str,
        target2: str,
        filter1: Optional[str],
        filter2: Optional[str],
        width: int = 64,
        height: int = 64,
        shift: Optional[int] = None,
        iterations: int = 5000,
        silent: bool = False
):
    """
    :param target1: input image1 path.
    :param target2: input image2 path.
    :param filter1: optionally output filter1 path.
    :param filter2: optionally output filter2 path.
    :param width: working image width in pixels.
    :param height: working image height in pixels.
    :param shift: shift to apply between filters in pixels.
    :param iterations: number of iterations.
    :param silent:  silence operation.
    :return:
    """

    shift = int(shift) if shift is not None else int(width / 20)
    target_image1 = load_image(target1, width=width, height=height)
    target_image2 = load_image(target2, width=width, height=height)
    A, B, pred1, pred2 = train_model(
        target_image1, target_image2,
        width=width, height=height, shift=shift,
        num_epochs=iterations,
        silent=silent
    )

    if filter1 is not None:
        filter_a = Image.fromarray(np.uint8(A[0, 0] * 255))
        filter_a.save(filter1)
    if filter2 is not None:
        filter_b = Image.fromarray(np.uint8(B[0, 0] * 255))
        filter_b.save(filter2)

    if not silent:
        # Visualize the learned matrices and predicted images
        visualize_results(A, B, pred1, pred2)


def main():
    CLI(evaluate)


if __name__ == '__main__':
    main()
