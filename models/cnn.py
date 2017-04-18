# -*- coding: utf-8 -*-
# @Author: Aman Priyadarshi
# @Date:   2017-04-17 18:18:50
# @Last Modified by:   amaneureka
# @Last Modified time: 2017-04-19 04:02:34

import numpy as np
import tensorflow as tf


def cnn_layer(input, num_channels, num_filters, filter_shape):

    # shape = [filter_height, filter_width, input_channels, output_channels]
    kernel_shape = [filter_shape[0], filter_shape[1], num_channels, num_filters]
    weights = tf.Variable(tf.truncated_normal(kernel_shape))
    biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer = tf.add(layer, biases)
    layer = tf.nn.avg_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 1, 1, 1],
                           padding='SAME')
    layer = tf.nn.tanh(layer)
    return layer, weights


def fc_layer(input, num_input, num_output):

    weights = tf.Variable(tf.truncated_normal(shape=[num_input, num_output]))
    biases = tf.Variable(tf.constant(0.05, shape=[num_output]))
    layer = tf.add(tf.matmul(input, weights), biases)
    return layer, weights


def create_network(img_height, img_width, num_classes):

    x = tf.placeholder(tf.float32, shape=[None, img_height * img_width])
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes])

    # input tensor
    tensor = tf.reshape(x, shape=[-1, img_height, img_width, 1])

    # two CNN layer
    layer1, w1 = cnn_layer(input=tensor,
                           num_channels=1,
                           num_filters=6,
                           filter_shape=(3, 3))
    layer2, w2 = cnn_layer(input=layer1,
                           num_channels=6,
                           num_filters=16,
                           filter_shape=(5, 5))

    # shape = [images, height, width, channels]
    features = layer2.get_shape()[1:].num_elements()
    layer2_flat = tf.reshape(layer2, shape=[-1, features])

    # two fully connected layers
    layer3, w3 = fc_layer(input=layer2_flat,
                          num_input=features,
                          num_output=50)
    layer3 = tf.nn.tanh(layer3)
    layer4, w4 = fc_layer(input=layer3,
                          num_input=50,
                          num_output=num_classes)
    y = tf.nn.softmax(layer4)

    # learning
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer4, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(cost)
    return x, y, y_true, optimizer
