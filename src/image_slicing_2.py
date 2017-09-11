'''
Created on 11 set 2017

@author: davide
'''

'''
    More complex example of image slicing 
'''

# Application to image processing, 2
# Split image into four corners

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import seaborn as sns
sns.set()

raw_image_data = mpimg.imread('./MarshOrchid.jpg')
print('raw_image_data.shape: ' + str(raw_image_data.shape))
print('raw_image_data.shape[0]//2: ' + str(raw_image_data.shape[0]//2))
print('raw_image_data.shape[1]//2: ' + str(raw_image_data.shape[1]//2))

image = tf.placeholder('uint8', [None, None, None], name = 'image')
sliceTopLeft = tf.slice(image, [0, 0, 0], [2674, 1842, -1])
sliceTopRight = tf.slice(image, [0, 1843, 0],
                         [2763, -1, -1])
sliceBottomLeft = tf.slice(image, [2753, 0, 0],
                         [-1, 1842, -1])
sliceBottomRight = tf.slice(image, [2674, 1843, 0],
                         [-1, -1, -1])

with tf.Session() as session:
    sliceTopLeftImage = session.run(sliceTopLeft, feed_dict = { image: raw_image_data})
    sliceTopRightImage = session.run(sliceTopRight, feed_dict = { image: raw_image_data})
    sliceBottomLeftImage = session.run(sliceBottomLeft, feed_dict = { image: raw_image_data})
    sliceBottomRightImage = session.run(sliceBottomRight, feed_dict = { image: raw_image_data})

plt.figure()
plt.imshow(sliceTopLeftImage)
plt.title('Top Left slice')
plt.show()

plt.figure()
plt.imshow(sliceTopRightImage)
plt.title('Top Right slice')
plt.show()

plt.figure()
plt.imshow(sliceBottomLeftImage)
plt.title('Bottom Left slice')
plt.show()

plt.figure()
plt.imshow(sliceBottomRightImage)
plt.title('Bottom Right slice')
plt.show()

plt.figure()
f, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(sliceTopLeftImage)
axarr[0, 0].set_title('Top Left')
axarr[0, 1].imshow(sliceTopRightImage)
axarr[0, 1].set_title('Top right')
axarr[1, 0].imshow(sliceBottomLeftImage)
axarr[1, 0].set_title('Bottom Left')
axarr[1, 1].imshow(sliceBottomRightImage)
axarr[1, 1].set_title('Bottom Right')
plt.show()
