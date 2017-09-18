#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  convert-to-TFRecords.py Author "Jinba Xiao <usar@npu-aslp.org>" Date 11.09.2017

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys

import numpy
import tensorflow as tf

import data_utils


tf.app.flags.DEFINE_string("cmvn_dir", "cmvn_dir2", "Cepstral Mean and Variance Normalization directory.")
tf.app.flags.DEFINE_string("data_dir", "data/scp", "Data directory.")
tf.app.flags.DEFINE_string("out_dir", "data/tfrecords", "Output data directory.")
tf.app.flags.DEFINE_boolean("compute_cmvn", False, "compute cmvn of training set")
tf.app.flags.DEFINE_boolean("apply_cmvn", True, "compute cmvn of training set")
tf.app.flags.DEFINE_integer("input_dim", 89, "Input dimension.")
tf.app.flags.DEFINE_integer("output_dim", 51, "Output dimension.")


FLAGS = tf.app.flags.FLAGS



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))



def convert_to_tfrecords(data_list, records_dir, label_dim, param_dim, apply_cmvn, label_cmvn, param_cmvn):
    if (not os.path.exists(records_dir)):
        os.mkdir(records_dir)

    for i in xrange(len(data_list)):
        label = numpy.fromfile(data_list[i][0], dtype=numpy.float32, count=-1, sep=' ')
        param = numpy.fromfile(data_list[i][1], dtype=numpy.float32, count=-1, sep=' ')
        row = len(label) // label_dim
        label = numpy.reshape(label, (row, label_dim))
        param = numpy.reshape(param, (row, param_dim))

        if (apply_cmvn):
            #label = (label - label_cmvn[0]) / label_cmvn[1]
            param = (param - param_cmvn[0]) / param_cmvn[1]
       
        records_filename = "%s/%s.tfrecord" % (records_dir, os.path.basename(data_list[i][0]).split('.')[0])
        print(records_filename)
        with tf.python_io.TFRecordWriter(records_filename) as writer:
            example = tf.train.SequenceExample(feature_lists=tf.train.FeatureLists(feature_list={
                'label':tf.train.FeatureList(feature=[_float_feature(input_) for input_ in label]),
                'param':tf.train.FeatureList(feature=[_float_feature(input_) for input_ in param])
            }))
            writer.write(example.SerializeToString())


def main(_):
    train_data, vali_data, test_data, label_cmvn, param_cmvn = data_utils.create_data(
                                    FLAGS.data_dir, FLAGS.cmvn_dir, FLAGS.compute_cmvn,
                                    FLAGS.input_dim, FLAGS.output_dim)
    if (not os.path.exists(FLAGS.out_dir)):
        os.mkdir(FLAGS.out_dir)

    convert_to_tfrecords(train_data, FLAGS.out_dir + '/train', FLAGS.input_dim, FLAGS.output_dim, FLAGS.apply_cmvn, label_cmvn, param_cmvn)
    convert_to_tfrecords(vali_data, FLAGS.out_dir + '/vali', FLAGS.input_dim, FLAGS.output_dim, FLAGS.apply_cmvn, label_cmvn, param_cmvn)
    convert_to_tfrecords(test_data, FLAGS.out_dir + '/test', FLAGS.input_dim, FLAGS.output_dim, FLAGS.apply_cmvn, label_cmvn, param_cmvn)

if __name__ == '__main__':
    tf.app.run()
