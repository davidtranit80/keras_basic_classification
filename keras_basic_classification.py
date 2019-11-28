# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow.keras as keras

# Helper libraries
import matplotlib as mpl

mpl.use('TkAgg')  # or whatever other backend that you want

# load data and divide dataset into training data and testing data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# scale image data from (0,255) to (0,1)
train_images = train_images / 255.0 
test_images = test_images / 255.0

# build a sequence model with two hidden layers with 64 nodes and one ouput layer with 10 nodes
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

# serialize model to JSON
model_json = model.to_json()
with open("basic_classification_model.json", "w") as json_file:
    json_file.write(model_json)
 
# serialize weights to HDF5
model.save_weights("basic_classification_weights.h5")
print("Saved model_basic_classification to disk")

# Predict test images
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
