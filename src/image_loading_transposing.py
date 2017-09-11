'''
Created on 11 set 2017

@author: davide
'''


# Image loading
import matplotlib.image as mpimg
import os

image = mpimg.imread('./../images/MarshOrchid.jpg')
print(image.shape)

import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()

# Transpose the image
import tensorflow as tf
x = tf.Variable(image, name = 'x')
model = tf.global_variables_initializer()
with tf.Session() as session:
    x = tf.transpose(x, perm = [1, 0, 2])
    session.run(model)
    result = session.run(x)
print(result.shape)
plt.figure()
plt.imshow(result)
plt.show()