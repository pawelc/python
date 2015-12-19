import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples