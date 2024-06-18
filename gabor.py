import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import conv2d

# Define parameters for Gabor filter
ksize = 5  # Kernel size
sigma = 3  # Standard deviation of the Gaussian envelope
theta = 1 * np.pi / 4  # Orientation of the normal to the parallel stripes of a Gabor function
lamda = 1 * np.pi / 4  # Wavelength of the sinusoidal factor
gamma = 0.4  # Spatial aspect ratio
phi = 0  # Phase offset

# Function to create Gabor kernel in PyTorch
def get_gabor_kernel(ksize, sigma, theta, lamda, gamma, phi):
    # Define the grid
    y, x = torch.meshgrid([torch.arange(-ksize // 2 + 1, ksize // 2 + 1, dtype=torch.float32), 
                           torch.arange(-ksize // 2 + 1, ksize // 2 + 1, dtype=torch.float32)])

    # Calculate Gabor kernel
    rotx = x * torch.cos(torch.tensor(theta)) + y * torch.sin(torch.tensor(theta))
    roty = -x * torch.sin(torch.tensor(theta)) + y * torch.cos(torch.tensor(theta))
    g = torch.exp(-0.5 * (rotx ** 2 / sigma ** 2 + roty ** 2 / (sigma / gamma) ** 2))
    g *= torch.cos(2 * np.pi * rotx / lamda + phi)
    
    return g

# Create Gabor kernel
kernel = get_gabor_kernel(ksize, sigma, theta, lamda, gamma, phi)

# Visualize the kernel
plt.imshow(kernel.numpy(), cmap='gray')
plt.title('Gabor Kernel')
plt.show()

# Load and process the image
img = cv2.imread(r'C:\Users\suraj\OneDrive\SOCmentee\assignment\Screenshot 2023-11-27 232315.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Convert image to PyTorch tensor and add batch and channel dimensions

# Apply Gabor filter using conv2d
kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions to kernel
filtered_img = conv2d(img, kernel)

# Convert filtered image back to numpy for visualization
filtered_img = filtered_img.squeeze().detach().numpy()

# Visualize the original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img.squeeze().numpy(), cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Filtered Image')
plt.imshow(filtered_img, cmap='gray')
plt.show()

# Resize and visualize the kernel
kernel_resized = cv2.resize(kernel.squeeze().numpy(), (400, 400))
cv2.imshow('Kernel', kernel_resized)
cv2.imshow('Original Img.', img.squeeze().numpy())
cv2.imshow('Filtered', filtered_img)
cv2.waitKey(5000)
cv2.destroyAllWindows()
