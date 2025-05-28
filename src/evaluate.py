import tensorflow as tf
(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
test_images = test_images / 255.0
test_images = test_images.reshape(-1, 28, 28, 1)

model = tf.keras.models.load_model("models/mnist_cnn.h5")
loss, acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {acc * 100:.2f}%")