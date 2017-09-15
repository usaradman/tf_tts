#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  run.py Author "Jinba Xiao <usar@npu-aslp.org>" Date 04.09.2017

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


tf.app.flags.DEFINE_string("train_dir", "cissy_5000_lstm", "Training directory.")
tf.app.flags.DEFINE_string("cmvn_dir", "cmvn_dir2", "Cepstral Mean and Variance Normalization directory.")
tf.app.flags.DEFINE_string("data_dir", "data/tfrecords", "Data directory.")
tf.app.flags.DEFINE_string("test_dir", "test", "Test output directory.")
tf.app.flags.DEFINE_boolean("decode", False, "Is decode otherwise training. default(False)")
tf.app.flags.DEFINE_boolean("compute_cmvn", False, "compute cmvn of training set")
tf.app.flags.DEFINE_integer("input_dim", 89, "Input dimension.")
tf.app.flags.DEFINE_integer("output_dim", 51, "Output dimension.")
tf.app.flags.DEFINE_integer("batch_size", 50, "Batch size.")

tf.app.flags.DEFINE_string("objective_function", "mse", "Objective function.")
tf.app.flags.DEFINE_integer("min_steps", 5, "Min training iteration.")
tf.app.flags.DEFINE_integer("max_steps", 100, "Max training iteration.")
tf.app.flags.DEFINE_integer("steps_ckpt", 1, "iteration num for save checkpoint.")
tf.app.flags.DEFINE_integer("keep_lr_steps", 3, "Data directory.")
tf.app.flags.DEFINE_float("learn_rate", 0.0001, "Data directory.")
tf.app.flags.DEFINE_float("momentum", 0.9, "Momentum. default(0.9)")
tf.app.flags.DEFINE_float("lr_decay_factor", 0.75, "Learning rate decay factor.")
tf.app.flags.DEFINE_float("start_decay_impr", 0.001, "Start decay learning rate when rel_impr < start_decay_impr.")
tf.app.flags.DEFINE_float("end_decay_impr", 0.0001, "Finish training when rel_impr < end_decay_impr.")


FLAGS = tf.app.flags.FLAGS

NNET_PROTO = [
                {'type':'DNN',
                'num_units':128,
                'activation':'relu',
                'dropout':False},
                {'type':'DNN',
                'num_units':128,
                'activation':'relu',
                'dropout':False},
                {'type':'LSTM',
                'num_units':128,
                'activation':'',
                'dropout':False}
          ]


def create_model(sess):
    '''
    Create NN model
    '''
    nnet_model = model.NnetModel(sess, NNET_PROTO,
                      FLAGS.input_dim,
                      FLAGS.output_dim,
                      FLAGS.batch_size,
                      FLAGS.learn_rate,
                      FLAGS.objective_function,
                      FLAGS.momentum)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        nnet_model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Creat model with initial parameters.")
        sess.run(tf.global_variables_initializer())
    return nnet_model

def train():
    '''
    Training
    '''
    train_data = [FLAGS.data_dir + '/train/' + input_ for input_ in os.listdir(FLAGS.data_dir + '/train/')]
    vali_data = [FLAGS.data_dir + '/vali/' + input_ for input_ in os.listdir(FLAGS.data_dir + '/vali/')]
    test_data = [FLAGS.data_dir + '/test/' + input_ for input_ in os.listdir(FLAGS.data_dir + '/test/')]

    if (not os.path.exists(FLAGS.train_dir)):
        os.mkdir(FLAGS.train_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        nnet_model = create_model(sess)

        print('Create model done, ready to train...')
        nnet_best = ''
        current_step = 0
        pre_vali_loss = nnet_model.step(vali_data)
        print('PRERUN AVG.LOSS %.4f' % pre_vali_loss)
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            train_loss = nnet_model.step(train_data)
            end_time = time.time()
            step_time = end_time - start_time
            current_step += 1

            if(math.isinf(train_loss) or math.isnan(train_loss)):
                print('Training error! train loss is NAN or INF')
                return

            vali_loss = nnet_model.step(vali_data)
            rel_impr = (pre_vali_loss - vali_loss) / pre_vali_loss

            if (current_step % FLAGS.steps_ckpt == 0):
                print('Iteration %0d : TRAIN AVG.LOSS %.4f, CROSSVAL AVG.LOSS %.4f, Impr %.4f, Learning Rate %.6f, Cost Time %.2f' % (current_step, train_loss, vali_loss, rel_impr, nnet_model.learn_rate, step_time))
                # Save checkpoint
                nnet_cur = 'model_step_%d.ckpt' % current_step
                checkpoint_path = os.path.join(FLAGS.train_dir, nnet_best)
                nnet_model.saver.save(sess, checkpoint_path, global_step=nnet_model.global_step)

                # gen test examples
                cur_test_dir = FLAGS.test_dir + '_Iter%0d' % (current_step)
                if (not os.path.exists(cur_test_dir)):
                    os.mkdir(cur_test_dir)
                nnet_model.test(test_data, cur_test_dir)

                if (current_step > FLAGS.max_steps):
                    print('Finished max steps.')
                    break

                if (vali_loss < pre_vali_loss or current_step <= FLAGS.keep_lr_steps or current_step <= FLAGS.min_steps):
                    nnet_best = nnet_cur
                    print('nnet accepted')
                else:
                    print('nnet rejected')

                if (current_step <= FLAGS.keep_lr_steps):
                    continue

                if (rel_impr < FLAGS.end_decay_impr):
                    if (current_step <= FLAGS.min_steps):
                        print('We were  supposed to Finish, but continue as min iter %d' % FLAGS.min_steps)
                    else:
                        print('Finished, too small rel. improvement %.6f' % rel_impr)
                    break

                if (rel_impr < FLAGS.start_decay_impr):
                    print('Too small improvement, learning rate decay')
                    nnet_model.decay_learn_rate(FLAGS.lr_decay_factor)

            pre_vali_loss = vali_loss




def decode():
    '''
    Decode
    '''


def main(_):
    if FLAGS.decode:
        decode()
    else:
        train()

if __name__ == '__main__':
    tf.app.run()
