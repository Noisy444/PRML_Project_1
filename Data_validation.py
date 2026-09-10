# Import Fashion-MNIST dataset from TensorFlow
from tensorflow.keras.datasets import fashion_mnist
# Import Matplotlib for displaying images
import matplotlib.pyplot as plt
# Import NumPy for numerical operations
import numpy as np

# Load the training and testing data
# x_train and x_test contain image data
# y_train and y_test contain labels for image data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Display the shapes of the image datasets
print(x_train.shape)
print(x_test.shape)

# Convert the image values to floating-point numbers
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten the 28x28 images into 1D arrays of 784 features 
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Define names of the ten clothing categories
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Display the first 10 images
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(classes[y_train[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()

# Set NumPy print options for better readability of floating-point numbers
np.set_printoptions(precision=3, floatmode='fixed')

# Display the pixel matrix of the third image
print(x_train[2])
print(x_train[2][:5, :10])

# Check the shape of the datasets
print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)

# Check the data type
print("Image data type:", x_train.dtype)

# Check pixel value range
print("Minimum pixel value:", x_train.min())
print("Maximum pixel value:", x_train.max())
print("Minimum pixel value (test set):", x_test.min())
print("Maximum pixel value (test set):", x_test.max())

plt.figure(figsize=(10, 5))

class_counts = np.bincount(y_train, minlength=10)
plt.bar(classes, class_counts)

plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Fashion-MNIST Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()