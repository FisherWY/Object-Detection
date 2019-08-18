from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow
import os
import config as cfg
import tensorflow as tf
from anchor_label import anchor_labels_process
from rois_target import proposal_target
from anchor_generate import all_anchor_conner
import cv2
import copy
slim = tf.contrib.slim
import numpy as np
import tfplot as tfp


def vgg16(self, input_image):
    with tf.variable_scope('vgg_16'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.repeat(input_image, 2, slim.conv2d, 64, [3, 3], trainable=False, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    return net


def rpn_net(self, input_feature_map, num_anchor):  # rpn net built
        with tf.variable_scope('rpn'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=tf.nn.relu, \
                                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), \
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                rpn_feature = slim.conv2d(input_feature_map, 512, [3, 3], scope='conv6')
                if self.is_training:
                    self.add_heatmap(rpn_feature, name='rpn_feature')
                rois_cls = slim.conv2d(rpn_feature, 2 * num_anchor, [1, 1], padding='VALID', activation_fn=None,
                                       scope='conv7')
                rois_reg = slim.conv2d(rpn_feature, 4 * num_anchor, [1, 1], padding='VALID', activation_fn=None,
                                       scope='conv8')
        return {'rois_cls': rois_cls, 'rois_bbx': rois_reg}

