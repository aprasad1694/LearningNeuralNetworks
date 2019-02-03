import tensorflow as tf

####        tensorflow programs usually have two pars:          #####
# 1. Building the computation graph -- Construction phase
#### a) Typically builds computation graph that represents the ML model and computations required
####    to train it

# 2. Running the computation graph -- execution phase
#### a) Generally runs a loop that evaluates a training step repeatedly (ex: one step per mini-batch)
####    gradually improving the model parameters

##################################################################################################

###### CONSTRUCTION PHASE ###########
### the three lines below does not computation -- it just creates a computation graph
# the variables (x and y) are not initialized yet
x = tf.Variable(3, name = "x")
y = tf.Variable(4, name = "y")
f = x*x*y + y + 2


###### EXECUTION PHASE ###########
### to evaluate graph above we need to open a tensorflow session which will....
# innitialize the variables
# evaluate f

# this is the long and verbose way of evaulating graph
"""
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
"""
# this is a less verbose way to do the saeme thing - also closes session automatically at end of block
"""
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
"""

# to do it even less verbosely, initialize all variables at once using tf.global_variables_initializer()
init = tf.global_variables_initializer() #gets all the nodes ready (not initiali until its run)

with tf.Session() as sess:
    init.run() #actually intitializes the variables
    result = f.eval()

print(result)

sess.close()

#### How long are node values stored for? How does that process work
# 1. All node values are dropped between graph runs except variable values
# 2. Variable values are maintained by the session across graph runs
# 3. A variable starts its life when its initializer is run and ends when session is closed

#for the code below w and x are calculated two times for each evaluation even in the execution phase

## construction phase
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

## execution phase
# here w and x are evaluated twice -- once when eval'ing y and once for z (since both depend on w and x)
with tf.Session() as sess:
    print(y.eval())
    print(z.eval())

# this is the way to evalute y and z without evaluating w and x twice:
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)
