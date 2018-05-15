import numpy as np
import tensorflow as tf

sess = tf.Session()

def print_t(T):
    print(T.eval(session=sess))