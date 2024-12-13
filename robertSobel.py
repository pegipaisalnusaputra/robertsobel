import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Fungsi untuk konversi gambar ke grayscale
def rgb2gray(image):
    if len(image.shape) == 3:
        return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    return image

# Operator Robert
def roberts_operator(image):
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    edge_x = convolve(image, kernel_x)
    edge_y = convolve(image, kernel_y)
    return np.sqrt(edge_x**2 + edge_y**2)

# Operator Sobel
def sobel_operator(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    edge_x = convolve(image, kernel_x)
    edge_y = convolve(image, kernel_y)
    return np.sqrt(edge_x**2 + edge_y**2)

# Load gambar
image = imageio.imread('penguinn.jpeg')  # Ganti dengan nama file gambar Anda
gray_image = rgb2gray(image)

# Deteksi tepi
edges_roberts = roberts_operator(gray_image)
edges_sobel = sobel_operator(gray_image)

# Plot hasil
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Robert Operator")
plt.imshow(edges_roberts, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Sobel Operator")
plt.imshow(edges_sobel, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
