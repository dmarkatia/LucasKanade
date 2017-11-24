# ==============================================================================
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the CNN exercise.
#  You will need to complete code in train_cnn.py
#
# ======================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import gzip
import pickle

import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

from train_cnn import ConvNet


# Set parameters for Sparse Autoencoder
parser = argparse.ArgumentParser('CNN Exercise.')
parser.add_argument('--learning_rate',
                    type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--num_epochs',
                    type=int,
                    default=60,
                    help='Number of epochs to run trainer.')
parser.add_argument('--decay',
                    type=float,
                    default=0.1,
                    help='Decay rate of l2 regularization.')
parser.add_argument('--batch_size',
                    type=int, default=10,
                    help='Batch size. Must divide evenly into the dataset sizes.')
parser.add_argument('--input_data_dir',
                    type=str,
                    default='../mnist/data',
                    help='Directory to put the training data.')
parser.add_argument('--expanded_data',
                    type=str,
                    default='../mnist/data/mnist_expanded.pkl.gz',
                    help='Directory to put the extended mnist data.')
parser.add_argument('--log_dir',
                    type=str,
                    default='logs',
                    help='Directory to put logging.')
parser.add_argument('--visibleSize',
                    type=int,
                    default=str(28 * 28),
                    help='Used for gradient checking.')
parser.add_argument('--hiddenSize',
                    type=int,
                    default='100',
                    help='.')

FLAGS = None
FLAGS, unparsed = parser.parse_known_args()
mode = int(sys.argv[1])

# ======================================================================
#  STEP 0: Load data from the MNIST database.
#  This loads our training and test data from the MNIST database files.
#  We have sorted the data for you in this so that you will not have to
#  change it.

data_sets = input_data.read_data_sets(FLAGS.input_data_dir)

# ======================================================================
#  STEP 1: Train a baseline model.
#  This trains a feed forward neural network with one hidden layer.
#  Expected accuracy >= 97.80%

if mode == 1:
  cnn = ConvNet(1)
  accuracy = cnn.train_and_evaluate(FLAGS, data_sets.train, data_sets.test)

  # Output accuracy.
  print(20 * '*' + 'model 1' + 20 * '*')
  print('accuracy is %f' % (accuracy))
  print()



# ======================================================================
#  STEP 2: Use two convolutional layers.
#  Expected accuracy >= 99.06%

if mode == 2:
  cnn = ConvNet(2)
  accuracy = cnn.train_and_evaluate(FLAGS, data_sets.train, data_sets.test)

  # Output accuracy.
  print(20 * '*' + 'model 2' + 20 * '*')
  print('accuracy is %f' % (accuracy))
  print()


# ======================================================================
#  STEP 3: Replace sigmoid activation with ReLU.
#
#  Expected accuracy>= 99.23%

if mode == 3:
  FLAGS.learning_rate = 0.03
  cnn = ConvNet(3)
  accuracy = cnn.train_and_evaluate(FLAGS, data_sets.train, data_sets.test)

  # Output accuracy.
  print(20 * '*' + 'model 3' + 20 * '*')
  print('accuracy is %f' % (accuracy))
  print()


# ======================================================================
#  STEP 4: Add one more fully connected layer.
#
#  Expected accuracy>= 99.37%

if mode == 4:
  FLAGS.learning_rate = 0.03
  cnn = ConvNet(4)
  accuracy = cnn.train_and_evaluate(FLAGS, data_sets.train, data_sets.test)

  # Output accuracy.
  print(20 * '*' + 'model 4' + 20 * '*')
  print('accuracy is %f' % (accuracy))
  print()


# ======================================================================
#  STEP 5: Add dropout to reduce overfitting.
#
#  Expected accuracy: 99.40%

if mode == 5:
  FLAGS.learning_rate = 0.03
  FLAGS.num_epochs    = 40
  FLAGS.hiddenSize    = 1000
  cnn = ConvNet(5)
  accuracy = cnn.train_and_evaluate(FLAGS, data_sets.train, data_sets.test)

  # Output accuracy.
  print(20 * '*' + 'model 5' + 20 * '*')
  print('accuracy is %f' % (accuracy))
  print()
