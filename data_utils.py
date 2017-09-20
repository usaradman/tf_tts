#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  data.py Author "Jinba Xiao <usar@npu-aslp.org>" Date 07.09.2017

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os

import numpy
from six.moves import xrange
import tensorflow as tf


def parse_function(data, input_dim, output_dim):
    features = {"label":tf.FixedLenSequenceFeature([input_dim], tf.float32),
                "param":tf.FixedLenSequenceFeature([output_dim], tf.float32)}
    _, parsed_features = tf.parse_single_sequence_example(data, sequence_features=features)
    input = parsed_features["label"]
    output = parsed_features["param"]
    row = tf.shape(input)[0]
    return input, output, row



def get_batch(data_list, batch_size, input_dim, output_dim, shuffle=True, repeat=1):
        dataset = tf.contrib.data.TFRecordDataset(data_list)
        dataset = dataset.map(lambda filename: parse_function(filename,input_dim, output_dim), num_threads=3)
        dataset = dataset.repeat(repeat)
        if (shuffle):
            dataset = dataset.shuffle(buffer_size=len(data_list) - 10)
        dataset = dataset.padded_batch(batch_size, padded_shapes=([None, input_dim], [None, output_dim], []))
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()



def read_file(filename, dim):
    feature = numpy.fromfile(filename, dtype=numpy.float32, count=-1, sep=' ')
    return numpy.reshape(feature, (len(feature)//dim, dim))


def _compute_cmvn(data, label_dim, param_dim, out_dir=''):
    '''
    CMVN
    '''
    data_size = len(data)
    label_mean = numpy.zeros(label_dim, dtype=numpy.float32)
    label_var = numpy.zeros(label_dim, dtype=numpy.float32)
    param_mean = numpy.zeros(param_dim, dtype=numpy.float32)
    param_var = numpy.zeros(param_dim, dtype=numpy.float32)

    # CMVN
    num_frames = 0
    for i in xrange(data_size):
        label = read_file(data[i][0], label_dim)
        param = read_file(data[i][1], param_dim)
        cur_frames = label.shape[0]
        num_frames += cur_frames

        for dim in xrange(label_dim):
            label_mean[dim] += numpy.sum(label[:,dim])
            label_var[dim] += numpy.sum(label[:,dim] * label[:,dim])

        for dim in xrange(param_dim):
            param_mean[dim] += numpy.sum(param[:,dim])
            param_var[dim] += numpy.sum(param[:,dim] * param[:,dim])

        '''
        for j in xrange(cur_frames):
            label_mean += label[j]
            label_var += label[j] * label[j]
            param_mean += param[j]
            param_var += param[j] * param[j]
        '''

    label_mean = label_mean / num_frames
    param_mean = param_mean / num_frames

    label_var = label_var / num_frames - label_mean * label_mean
    param_var = param_var / num_frames - param_mean * param_mean

    label_var[label_var < 1.0e-20] = 1.0e-20
    param_var[param_var < 1.0e-20] = 1.0e-20

    label_var = numpy.sqrt(label_var)
    param_var = numpy.sqrt(param_var)

    label_cmvn = numpy.vstack([label_mean, label_var])
    param_cmvn = numpy.vstack([param_mean, param_var])

    if(out_dir.strip() == ''):
        out_dir = 'cmvn_dir'

    if (not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    numpy.savetxt(out_dir + '/label_cmvn', label_cmvn, fmt='%f')
    numpy.savetxt(out_dir + '/param_cmvn', param_cmvn, fmt='%f')

    return label_cmvn, param_cmvn



def get_cmvn(dir):
    label_cmvn = numpy.fromfile(dir + '/label_cmvn', dtype=numpy.float32, count=-1, sep=' ')
    param_cmvn = numpy.fromfile(dir + '/param_cmvn', dtype=numpy.float32, count=-1, sep=' ')
    label_cmvn = numpy.reshape(label_cmvn, (2, len(label_cmvn) // 2))
    param_cmvn = numpy.reshape(param_cmvn, (2, len(param_cmvn) // 2))
    return label_cmvn, param_cmvn


def apply_cmvn(input, mean, var):
    return input - mean / var


def create_data(data_dir, cmvn_dir, compute_cmvn, label_dim, param_dim):
    '''
    Create data for NN
    [ 'label_filename', 'param_filename']
    '''
    train_data = None
    vali_data = None
    test_data = None
    for data in ['train', 'val', 'test']:
        label_filename = os.path.join(data_dir, 'label_' + data + '.scp')
        param_filename = os.path.join(data_dir, 'param_' + data + '.scp')
        label_fp = open(label_filename, 'r')
        param_fp = open(param_filename, 'r')
        label_lines = label_fp.readlines()
        param_lines = param_fp.readlines()
        label_fp.close()
        param_fp.close()

        print('Read data from %s len: %d' % (label_filename, len(label_lines)))

        if (data == 'train'):
            train_data = [ ['', ''] for _ in xrange(len(label_lines))]
            #train_data = numpy.ones([len(label_lines), 2]).astype(numpy.str)
            for i in xrange(len(label_lines)):
                train_data[i][0] = label_lines[i].split(' ')[1].strip()
                train_data[i][1] = param_lines[i].split(' ')[1].strip()
        elif (data == 'val'):
            vali_data = [ ['', ''] for _ in xrange(len(label_lines))]
            #vali_data = numpy.ones([len(label_lines), 2]).astype(numpy.str)
            for i in xrange(len(label_lines)):
                vali_data[i][0] = label_lines[i].split(' ')[1].strip()
                vali_data[i][1] = param_lines[i].split(' ')[1].strip()
        else:
            test_data = [ ['', ''] for _ in xrange(len(label_lines))]
            #test_data = numpy.ones([len(label_lines), 2]).astype(numpy.str)
            for i in xrange(len(label_lines)):
                test_data[i][0] = label_lines[i].split(' ')[1].strip()
                test_data[i][1] = param_lines[i].split(' ')[1].strip()
    label_cmvn = None
    param_cmvn = None
    if (compute_cmvn):
        print('Computing CMVN...')
        label_cmvn, param_cmvn = _compute_cmvn(train_data, label_dim, param_dim, cmvn_dir)
    else:
        label_cmvn, param_cmvn = get_cmvn(cmvn_dir)
    return train_data, vali_data, test_data, label_cmvn, param_cmvn


