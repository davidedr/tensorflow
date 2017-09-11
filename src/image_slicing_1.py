'''
Created on 11 set 2017

@author: davide
'''

'''
    Example use of slice operator
'''

# Application to image processing
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import seaborn as sns
sns.set()

raw_image = mpimg.imread('./MarshOrchid.jpg')

image = tf.placeholder('uint8', [None, None, 3], name = 'image')
slice = tf.slice(image, [1000, 0, 0], [3000, -1, -1])
with tf.Session() as session:
    sliced_image = session.run(slice, feed_dict = { image: raw_image })

print('raw_image.shape: ' + str(raw_image.shape))
plt.imshow(raw_image)
plt.title('Raw image')
plt.show()

print('sliced_image.shape: ' + str(sliced_image.shape))
plt.imshow(sliced_image)
plt.title('Sliced image')
plt.show() 