'''
Created on 11 set 2017

@author: davide
'''

'''
    A loop to compute moving average
'''

# loop example: moving average
import tensorflow as tf
import numpy as np
x = tf.Variable(0, name = 'x')
n = 0
with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("graph", session.graph)

    model = tf.global_variables_initializer()
    session.run(model)
    
    for i in range(10):
        y = np.random.randint(1E3)
        x = (n*x + y)
        n += 1
        x = x/n
        print(session.run(x))
