from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow as tf

# run code in gan_env
mnist = read_data_sets("mnist", one_hot=True)
print(mnist)