import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
  mean = [0,3],
  cov = [[1,0.5],[0.5,1]],
  size = num_samples_per_class
)
positive_samples = np.random.multivariate_normal(
  mean = [3,0],
  cov = [[1,0.5], [0.5,1]],
  size= num_samples_per_class
)

inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

targets = np.vstack((np.zeros((num_samples_per_class,1), dtype="float32"),
                     np.ones((num_samples_per_class, 1), dtype="float32")))


input_dim = 2
output_dim = 1

w = tf.Variable(initial_value= tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

def model(inputs):
  return tf.matmul(inputs,w) +b

def square_loss(targets, predictions):
  per_sample_losses = tf.square(targets-predictions)
  return tf.reduce_mean(per_sample_losses)


learning_rate = 0.1

def training_step(inputs, targets):
  with tf.GradientTape() as tape: 
    predictions = model(inputs)
    loss= square_loss(targets, predictions)
  grad_loss_wrt_w, grad_loss_wrt_b = tape.gradient(loss, [w,b])
  w.assign_sub(grad_loss_wrt_w * learning_rate)
  b.assign_sub(grad_loss_wrt_b * learning_rate)
  return loss
  
  
for step in range(40):
  loss= training_step(inputs, targets)
  print (f"loss at step{step}: {loss:.4f}")
  
  
predictions = model(inputs)

x = np.linspace(-1,4, 100)
y=  -w[0]/w[1]*x+(0.5-b)/w[1]
plt.plot(x,y,"-r")
plt.scatter(inputs[:,0], inputs[:,1], c=predictions[:,0] > 0.5)
plt.show()

print (np.size(x))
print(np.size(y))
