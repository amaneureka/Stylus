# -*- coding: utf-8 -*-
# @Author: Aman Priyadarshi
# @Date:   2017-04-17 18:18:50
# @Last Modified by:   amaneureka
# @Last Modified time: 2017-05-19 05:02:30

import numpy as np
import tensorflow as tf

def cnn_layer(input, num_channels, num_filters, filter_shape):

		# shape = [filter_height, filter_width, input_channels, output_channels]
		kernel_shape = [filter_shape[0], filter_shape[1], num_channels, num_filters]
		weights = tf.Variable(tf.random_normal(kernel_shape))
		biases = tf.Variable(tf.random_normal(shape=[num_filters]))

		layer = tf.nn.conv2d(input=input,
								filter=weights,
								strides=[1, 1, 1, 1],
								padding='SAME')
		layer = tf.add(layer, biases)
		return layer, weights


def fc_layer(input, num_input, num_output):

		weights = tf.Variable(tf.random_normal(shape=[num_input, num_output]))
		biases = tf.Variable(tf.random_normal(shape=[num_output]))
		layer = tf.add(tf.matmul(input, weights), biases)
		return layer, weights

def inception2d(input, num_channels, num_filter):
		# bias dimension = 3*num_filter and then the extra num_channels for the avg pooling
		bias = tf.Variable(tf.random_normal([3*num_filter + num_channels]))

		# 1x1
		one_filter = tf.Variable(tf.random_normal([1, 1, num_channels, num_filter]))
		one_by_one = tf.nn.conv2d(input, one_filter, strides=[1, 1, 1, 1], padding='SAME')

		# 3x3
		three_filter = tf.Variable(tf.random_normal([3, 3, num_channels, num_filter]))
		three_by_three = tf.nn.conv2d(input, three_filter, strides=[1, 1, 1, 1], padding='SAME')

		# 5x5
		five_filter = tf.Variable(tf.random_normal([5, 5, num_channels, num_filter]))
		five_by_five = tf.nn.conv2d(input, five_filter, strides=[1, 1, 1, 1], padding='SAME')

		# avg pooling
		pooling = tf.nn.avg_pool(input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

		input = tf.concat([one_by_one, three_by_three, five_by_five, pooling], axis=3)  # Concat in the 4th dim to stack
		input = tf.nn.bias_add(input, bias)
		return tf.nn.relu(input)


def create_network(img_height, img_width, num_classes):

		x = tf.placeholder(tf.float32, shape=[None, img_height * img_width])
		y_true = tf.placeholder(tf.float32, shape=[None, num_classes])

		# input tensor
		tensor = tf.reshape(x, shape=[-1, img_height, img_width, 1])

		# first CNN layer
		layer, w1 = cnn_layer(input=tensor,
								num_channels=1,
								num_filters=16,
								filter_shape=(3, 3))
		layer = tf.nn.max_pool(value=layer,
								ksize=[1, 3, 3, 1],
								strides=[1, 1, 1, 1],
								padding='SAME')
		layer = tf.nn.relu(layer)

		# second CNN layer
		layer, w2 = cnn_layer(input=layer,
								num_channels=16,
								num_filters=10,
								filter_shape=(3, 3))
		layer = tf.nn.max_pool(value=layer,
								ksize=[1, 2, 2, 1],
								strides=[1, 1, 1, 1],
								padding='SAME')
		# layer = tf.nn.dropout(layer, 0.75)
		layer = tf.nn.relu(layer)

		# Inception Module
		layer = inception2d(input=layer,
								num_channels=10,
								num_filter=16)

		# shape = [images, height, width, channels]
		features = layer.get_shape()[1:].num_elements()
		layer = tf.reshape(layer, shape=[-1, features])

		# two fully connected layers
		layer, w4 = fc_layer(input=layer,
								num_input=features,
								num_output=500)
		layer = tf.nn.tanh(layer)
		layer, w6 = fc_layer(input=layer,
								num_input=500,
								num_output=num_classes)
		y = tf.nn.softmax(layer)

		# learning
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer, labels=y_true)
		cost = tf.reduce_mean(cross_entropy)
		optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0025).minimize(cost)
		return x, y, y_true, optimizer
