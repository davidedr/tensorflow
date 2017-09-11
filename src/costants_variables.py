'''
Created on 11 set 2017

@author: davide
'''

'''
    Basic use og constants and varables
'''
# Tensor flow basics

x = 35
y = x + 5
print(y)

# Tf equivalent
import tensorflow as tf
x = tf.constant(35, name = 'x')
y = tf.Variable(x + 5, name ='y')

# Not a value, a tensor flow object intead
print(y)

model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    y_value = session.run(y)
print(y_value)

# Array example
x = tf.constant([35, 40, 45], name = 'x')
y = tf.Variable(x + 5, name = 'y')

model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    y_value = session.run(y)
print(y_value)

#
import numpy as np
x = tf.constant(np.random.randint(0, high = 1E1+1, size = 2E1), name = 'x')
y = tf.Variable(5*x**2-3*x+15, name = 'y')

model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    y_value = session.run(y)
print(y_value)