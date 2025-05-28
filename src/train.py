import tensorflow as tf
from model import build_model

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
train_images = train_images.reshape(-1, 28, 28, 1)

model = build_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
model.save("models/mnist_cnn.h5")
