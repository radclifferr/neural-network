import tensorflow as tf
import math
from tensorflow.keras.datasets import mnist
import numpy as np
import random
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

class NaiveDense:
  #self is self, python stuff, 
  #input size is shape of tensor, 
  # output size is a tensor of one order less? I think
  #activation is a function like ReLU or... cant remember. 
  def __init__(self, input_size, output_size, activation):
    self.activation = activation

    #Shape of the weights for the layer
    w_shape = (input_size, output_size)
    # Set the initial value of the weights to be a random value between 0 and 1
    w_initial_value = tf.random.uniform(w_shape, minval = 0, maxval = 1e-1)
    #assign self.W and create it as a tensor flow Variable
    self.W = tf.Variable(w_initial_value)
    #Set the shape of the bias function
    b_shape = (output_size)
    #Set the inital values of the bias to be the shape of b populated as all zeros
    b_initial_value = tf.zeros(b_shape)
    #assign b to the class as whatever tv.Variable is I guess. 
    self.b = tf.Variable(b_initial_value)


  #output = activation(dot(W, input) + B)
  def __call__(self, inputs):
    # do a dot product of the inputs and the weight to apply weightages to the input,
    # then add the bias 
    #then, given a certian activation function, return a result. 
    return self.activation(tf.matmul(inputs, self.W) + self.b)
  @property
  def weights(self):
    return [self.W, self.b]


#Chains layers together
class NaiveSequential:
  def __init__(self, layers):
    self.layers = layers

  def __call__(self, inputs):
    x = inputs
    for layer in self.layers:
      x = layer(x)
    return x
  
  @property
  def weights(self):
    weights = []
    for layer in self.layers:
      weights += layer.weights
    return weights
  
## Create the AI Model with two layers by calling naive sequental class with two dense layers
  
model = NaiveSequential([
  NaiveDense(input_size = 28 * 28, output_size= 512, activation= tf.nn.relu),
  NaiveDense(input_size = 512, output_size= 10, activation = tf.nn.softmax)
])
assert len(model.weights) == 4

learning_rate = 1e-3

def update_weights(gradients, weights):
  for g,w in zip(gradients, weights):
    w.assign_sub(g * learning_rate)


def one_training_step(model, images_batch, labels_batch):
  with tf.GradientTape() as tape:
    predictions = model(images_batch)
    per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(labels_batch, predictions)
    average_loss = tf.reduce_mean(per_sample_losses)
  gradients = tape.gradient(average_loss, model.weights)
  update_weights(gradients, model.weights)
  return average_loss

## Create a batch generator to iterate over the test data in batches for learning

class BatchGenerator:
  def __init__(self, images, labels, batch_size = 128):
    #check of length is correct between image count and label count
    assert len(images) == len(labels)
    # set initial values,
    self.index = 0
    self.images = images
    self.labels = labels
    self.batch_size = batch_size
    # determine number of batches from batch size and number of images
    self.num_batches = math.ceil(len(images)/batch_size)

  def next(self):
    images = self.images[self.index : self.index + self.batch_size]
    labels = self.labels[self.index : self.index + self.batch_size]
    self.index += self.batch_size
    return images, labels


def fit(model, images, labels, epochs, batch_size = 128):
  for epoch_counter in range(epochs):
    print(f"Epoch{epoch_counter}")
    batch_generator = BatchGenerator(images, labels, batch_size)
    for batch_counter in range(batch_generator.num_batches):
      images_batch, labels_batch = batch_generator.next()
      loss = one_training_step(model, images_batch, labels_batch)
      if batch_counter % 100 == 0:
        print(f"Loss at batch{batch_counter}: {loss:.2f}")


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") /255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") /255

fit(model, train_images, train_labels, epochs = 10, batch_size = 128)

predictions = model(test_images)
predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels

for i in range(10000):
  random_number = random.randint(1, 10000) 
  if predicted_labels[random_number] == test_labels[random_number]:
    print("we have a match")
  else: 
    print("no match!")

print (len(matches))
print(f"accuracy: {matches.mean():.2f}")
