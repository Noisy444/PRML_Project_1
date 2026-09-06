# Import Fashion-MNIST dataset from TensorFlow
from tensorflow.keras.datasets import fashion_mnist
# Import Matplotlib for displaying images
import matplotlib.pyplot as plt

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

# Display the first image from the training dataset
plt.imshow(x_train[0], cmap="gray")

# Display the clothing category as the image title
# y_train[0] is the numeric label for the first image
plt.title(classes[y_train[0]])

# Show the image window
plt.show()