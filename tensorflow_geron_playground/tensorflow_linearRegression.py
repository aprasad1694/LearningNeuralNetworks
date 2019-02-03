import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

#### retreive the data
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

#### Preprocessing data for TF
# add a bias term (X0 = 1) for all training instances
m, n = housing.data.shape
scaler = StandardScaler()
housing.data = scaler.fit_transform(housing.data)
housing_data_plus_bias = np.c_[np.ones((m, 1)),
                               housing.data]
print("number of rows in housing.data: {m}".format(m=m))
print("number of rows in housing.data: {n}".format(n=n))


#### CONSTRUCTION PHASE
X = tf.constant(housing_data_plus_bias,
                dtype = tf.float32,
                name = "X")

y = tf.constant(housing.target.reshape(-1,1),
                dtype = tf.float32,
                name = "y")

# calculation of the normal equation -- the closed form solution for lin reg optimization
X_transpose = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(X_transpose, X)),X_transpose), y)


#### EXECUTION PHASE
with tf.Session() as sess:
    theta_value = theta.eval()
print(theta_value)
