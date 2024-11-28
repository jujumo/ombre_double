import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# External cost function to compute the loss
def compute_loss(pred1, pred2, target1, target2):
    # L2 loss (mean squared error)
    loss1 = torch.nn.functional.mse_loss(pred1, target1)
    loss2 = torch.nn.functional.mse_loss(pred2, target2)
    return loss1 + loss2


# Model class definition
class ImageReconstructionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define two learnable matrices A and B
        self.A = nn.Parameter(torch.randn(1, 1, 128, 128))  # Matrix A, 256x256 grayscale image
        self.B = nn.Parameter(torch.randn(1, 1, 128, 128))  # Matrix B, 256x256 grayscale image

    def forward(self, shift_x=2, shift_y=2):
        # Predict the first and second images based on A and B
        pred1 = self.A + self.B
        pred2 = self.A + torch.roll(self.B, shifts=(shift_y, shift_x), dims=(2, 3))

        return pred1, pred2


# Function to load and preprocess images (convert to grayscale and resize)
def load_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to 256x256 for consistency
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


# Main training loop
def train_model(target1, target2, num_epochs=500, learning_rate=1e-2):
    # Initialize model and optimizer
    model = ImageReconstructionModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
        pred1, pred2 = model()
        loss = compute_loss(pred1, pred2, target1, target2)
        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Return the learned matrices A and B
    return model.A.detach().numpy(), model.B.detach().numpy(), pred1.detach().numpy(), pred2.detach().numpy()


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


# Example usage
target_image1_path = 'target_image1.jpg'  # Path to the first target image
target_image2_path = 'target_image2.jpg'  # Path to the second target image
target1 = load_image(target_image1_path)
target2 = load_image(target_image2_path)
A, B, pred1, pred2 = train_model(target1, target2, num_epochs=5000)

# Visualize the learned matrices and predicted images
visualize_results(A, B, pred1, pred2)
