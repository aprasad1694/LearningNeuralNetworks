import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

#### retreive the data
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

#### Preprocessing data for TF
# add a bias term (X0 = 1) for all training instances (after scaling)
m, n = housing.data.shape
scaler = StandardScaler()
housing.data = scaler.fit_transform(housing.data)
housing_data_plus_bias = np.c_[np.ones((m, 1)),
                               housing.data]
print("number of rows in housing.data: {m}".format(m=m))
print("number of cols in housing.data: {n}".format(n=n))

##### CONSTRUCTION PHASE
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(housing_data_plus_bias,
                dtype = tf.float32,
                name = "X")
y = tf.constant(housing.target.reshape(-1,1),
                dtype = tf.float32,
                name = "y")
# random_uniform() creates a node in graph that generates a tensor containing random values given:
#       1. Shape: [n+1, 1] -- [1001,1]
#       2. Value range: (-1.0, 1.0)
theta = tf.Variable(tf.random_uniform([n+1,1], -1.0, 1.0), name = "theta")
y_pred = tf.matmul(X, theta, name = "predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name = "mse")
### TWO METHODS OF CALCULATING GRADIENT
# this method 'manually' calculates the gradient
# gradients = 2/m * tf.matmul(tf.transpose(X), error) 
# this method uses TensorFlow's reverse-mode audodiff to calc the gradients
gradients = tf.gradients(mse, [theta])[0]
# assign() creates a node that assigns a new value to a variable
# Here it implements the batch gradient decent step theta(next) = theta(current) - [learningRate * gradients]
training_op = tf.assign(theta, theta - learning_rate * gradients)

##### EXECUTION PHASE
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) # initialize all variables

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch {epoch} has MSE = {mse}".format(epoch = epoch,
                                                         mse = mse.eval()))
            print(theta.eval())
        sess.run(training_op) #for each epoch we reassign the value of theta

    best_theta = theta.eval()

print(best_theta)
