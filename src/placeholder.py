'''
Created on 11 set 2017

@author: davide
'''

'''
    Placeholder example
'''

# Placeholder example
import tensorflow as tf

x = tf.placeholder('float', None)
y = x * 2
with tf.Session() as session:
    result = session.run(y, feed_dict = { x: [1, 2, 3] })
print(result)

print()
x_data = [[1, 2, 3], [4, 5, 6]]
with tf.Session() as session:
    result = session.run(y, feed_dict = { x: x_data })
print(result)

# Placeholder example, 2
import tensorflow as tf

# Fixing the number of cols
x = tf.placeholder('float', [None, 3])
y = x * 2

x_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
with tf.Session() as session:
    result = session.run(y, feed_dict = { x: x_data })
print(result)