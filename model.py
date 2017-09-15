#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  model.py Author "Jinba Xiao <usar@npu-aslp.org>" Date 04.09.2017

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import time

import numpy
from six.moves import xrange
import tensorflow as tf

import data_utils



class NnetModel(object):
    '''
    NnetModel
    This class implements a multi-layer neural network
    input : [batch_size, max_seq_length, input_dim]
    '''

    def __init__(self, sess, proto, input_dim, output_dim, batch_size, learn_rate, objective_function, momentum):
        self.sess = sess
        self.proto = proto
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.momentum = momentum
        self.objective_function = objective_function

         
        self.cur_batch_size = 1
        self.global_step = tf.Variable(0, tf.int32)
        self.seq_length = tf.placeholder(tf.int32, shape=[None])
        self.train_input = tf.placeholder(tf.float32, shape=[self.batch_size, None, self.input_dim])
        self.decode_input = tf.placeholder(tf.float32, shape=[1, None, self.input_dim])
        self.train_output = tf.placeholder(tf.float32, shape=[None, None, self.output_dim])
        self.train_output_t = tf.reshape(self.train_output, [-1, self.output_dim])

        with tf.variable_scope('NNet_model') as scope:
            self.logits = self._build(self.proto, self.train_input, self.seq_length)
            scope.reuse_variables()
            self.decode_logits = self._build(self.proto, self.decode_input, self.seq_length)

        if (self.objective_function == 'mse'):
            self.loss = self._loss(self.train_output_t, self.logits, self.seq_length, self.cur_batch_size)
            self.decode_loss = self._loss(self.train_output_t, self.decode_logits, self.seq_length, self.cur_batch_size)

        self.train_op = tf.train.MomentumOptimizer(self.learn_rate, self.momentum).minimize(self.loss, global_step=self.global_step)
        
        self.saver = tf.train.Saver(tf.global_variables())



    def _build(self, proto, nnet_input, seq_length):
        '''
        Build graph
        '''
        last_output = nnet_input
        layer_index = 0
        for layer in proto:
            layer_index += 1
            print('Create layer:%d type:%s num_units:%d activation:%s ' % (layer_index, layer['type'], layer['num_units'], layer['activation']))
            if (layer['type'] == 'LSTM' or layer['type'] == 'GRU' or layer['type'] == 'RNN'):
                with tf.variable_scope('RNN_%d' % layer_index):
                    rnn_output, _ = self._RNN(layer['num_units'], layer['type'], layer['dropout'], layer['activation'], last_output, seq_length)
            elif (layer['type'] == 'DNN'):
                with tf.variable_scope('DNN_%d' % layer_index):
                    last_output = self._DNN(layer['num_units'], layer['dropout'], layer['activation'], last_output)
            else:
                continue

        with tf.variable_scope('OUTPUT'):
            last_output = self._DNN(self.output_dim, False, '', last_output)
        dim = tf.cast(last_output.shape[-1], tf.int32)
        last_output = tf.reshape(last_output, [-1, dim])
        return last_output


    def step(self, dataset, training=False):
        '''
        one epoch
        '''
        if (training):
            next_batch = data_utils.get_batch(dataset, self.batch_size, self.input_dim, self.output_dim)
        else:
            next_batch = data_utils.get_batch(dataset, 1, self.input_dim, self.output_dim)

        #for ele in tf.trainable_variables():
        #    print(ele.name)

        loss = 0
        while True:
            try:
                input_data, output_data, seq_length = self.sess.run(next_batch)
                self.cur_batch_size = input_data.shape[0]
                if (training):
                    if (self.cur_batch_size != self.batch_size):
                        continue
                    pre_loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.train_input:input_data, self.train_output:output_data,
                                                        self.seq_length:seq_length})
                    
                    #self.sess.run(self.train_op, feed_dict={self.train_input:input_data, self.train_output:output_data,
                    #                                    self.seq_length:seq_length})
                    loss = self.sess.run(self.loss, feed_dict={self.train_input:input_data, self.train_output:output_data,
                                                        self.seq_length:seq_length})
                    print('bach: %d preloss %.4f loss: %.4f' % (input_data.shape[0], pre_loss, loss))
                else:
                    loss += self.sess.run(self.decode_loss, feed_dict={self.decode_input:input_data, self.train_output:output_data,
                                                        self.seq_length:seq_length})

            except tf.errors.OutOfRangeError:
                if (not training):
                    loss = loss / len(dataset)
                break

        return loss


    def test(self, dataset, out_dir):
        next_batch = data_utils.get_batch(dataset, 1, self.input_dim, self.output_dim)
        ind = 0
        while True:
            try:
                input_data, _, seq_length = self.sess.run(next_batch)
                self.cur_batch_size = input_data.shape[0]
                output = self.sess.run(self.decode_logits, feed_dict={self.decode_input:input_data,
                                                        self.seq_length:seq_length})
                output = output[:seq_length[0]]
                filename = os.path.basename(dataset[ind]).split('.')[0] + '.cmp'
                numpy.savetxt(out_dir + '/' + filename, output, fmt='%f')
                ind += 1
            except tf.errors.OutOfRangeError:
                break


    def _loss(self, output, logits, seq_length, batch_size):
        '''
        compute loss
        '''
        diff = tf.Variable(0, dtype=tf.float32)
        num_frames = 0
        for i in xrange(batch_size):
            diff += tf.reduce_sum(tf.pow(output[i * batch_size : i * batch_size + seq_length[i]]
                                          - logits[i * batch_size : i * batch_size + seq_length[i]], 2))
            num_frames += seq_length[i]
        #diff *= 0.5
        diff = diff / tf.cast(num_frames,tf.float32)  
        return diff



    def decay_learn_rate(self, decay_factor):
        self.learn_rate *= decay_factor


    def _RNN(self, num_units, cell = 'LSTM', dropout = False, activation = '', input = None, seq_length = None):
        rnn_cell = None
        if (cell == 'LSTM'):
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units)
        elif (cell == 'GRU'):
            rnn_cell = tf.contrib.rnn.GRUCell(num_units)
        else:
            rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units)

        output, state = tf.nn.dynamic_rnn(rnn_cell, input, sequence_length=seq_length, dtype=tf.float32)
        return output, state


    def _DNN(self, num_units, dropout = False, activation = 'relu', input = None):
        #size = tf.shape(input)[0]
        size = input.get_shape().as_list()[0]
        input = tf.reshape(input, [-1, tf.cast(input.shape[-1], tf.int32)])
        weights = tf.get_variable('weights', [input.shape[-1], num_units], initializer=tf.random_normal_initializer(), dtype=tf.float32)
        biases = tf.get_variable('biases', [num_units], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        if (activation == 'relu'):
            output = tf.nn.relu(tf.matmul(input, weights) + biases)
        elif (activation == 'tanh'):
            output = tf.nn.tanh(tf.matmul(input, weights) + biases)
        elif (activation == 'sigmoid'):
            output = tf.nn.sigmoid(tf.matmul(input, weights) + biases)
        else:
            output = tf.matmul(input, weights) + biases
        #output = tf.reshape(output, [size, -1, num_units])
        output = tf.stack(tf.split(output, size, axis=0))
        return output


    def _CONV(self, num_units, dropout, input):
        '''
        Not implemented
        '''
