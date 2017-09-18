#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  convert-to-TFRecords-test.py Author "Jinba Xiao <usar@npu-aslp.org>" Date 12.09.2017

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import sys
import time
import math

import numpy
import tensorflow as tf

import model
import data_utils


tf.app.flags.DEFINE_string("train_dir", "cissy_1900", "Training directory.")
tf.app.flags.DEFINE_string("cmvn_dir", "cmvn_dir2", "Cepstral Mean and Variance Normalization directory.")
tf.app.flags.DEFINE_string("data_dir", "data/tfrecords", "Data directory.")
tf.app.flags.DEFINE_string("test_dir", "test", "Test output directory.")
tf.app.flags.DEFINE_boolean("compute_cmvn", False, "compute cmvn of training set")
tf.app.flags.DEFINE_integer("input_dim", 89, "Input dimension.")
tf.app.flags.DEFINE_integer("output_dim", 51, "Output dimension.")

FLAGS = tf.app.flags.FLAGS

def test(sess, dataset, out_dir, input_dim, output_dim, apply_cmvn=False, param_cmvn=None):
        next_batch = data_utils.get_batch(dataset, 1, input_dim, output_dim, shuffle=False)
        ind = 0
        while True:
            try:
                input, output, seq_length = sess.run(next_batch)
                output = numpy.reshape(output, (-1, output_dim))
                output = output[:seq_length[0]]
                print(seq_length[0])
                if(apply_cmvn):
                    output = output * param_cmvn[1] + param_cmvn[0]

                filename = os.path.basename(dataset[ind]).split('.')[0] + '.cmp'
                numpy.savetxt(out_dir + '/' + filename, output, fmt='%f')
                print('write one %s' % filename)
                ind += 1
            except tf.errors.OutOfRangeError:
                break

if __name__ == '__main__':
    test_data = [FLAGS.data_dir + '/test/' + input_ for input_ in os.listdir(FLAGS.data_dir + '/test/')]
    label_cmvn, param_cmvn = data_utils.get_cmvn(FLAGS.cmvn_dir)
    with tf.Session() as sess:
        # gen test examples
        cur_test_dir = FLAGS.test_dir
        if (not os.path.exists(cur_test_dir)):
            os.mkdir(cur_test_dir)
        test(sess, test_data, cur_test_dir, FLAGS.input_dim ,FLAGS.output_dim, True, param_cmvn)


