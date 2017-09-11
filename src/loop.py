'''
Created on 11 set 2017

@author: davide
'''

'''
    Loop example
'''

# Loop example
import tensorflow as tf

x = tf.Variable(0, name = 'x')
model = tf.global_variables_initializer()
print(model)
with tf.Session() as session:
    session.run(model)
    for i in range(5):
        x = x + 1
        print(session.run(x))