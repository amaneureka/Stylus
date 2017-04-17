# -*- coding: utf-8 -*-
# @Author: Aman Priyadarshi
# @Date:   2017-04-17 11:39:30
# @Last Modified by:   Aman Priyadarshi
# @Last Modified time: 2017-04-17 18:05:27

import numpy as np
import tensorflow as tf


def load_training_data(filepath, image_flat_size, total_classes, samples_per_class):

	img = np.fromfile(filepath, dtype=np.uint8)
	img.shape = (-1, image_flat_size)
	onehot = np.zeros(img.shape[0])
	onehot.shape = (-1, samples_per_class)
	for i in xrange(total_classes):
		onehot[i, ] = i
	return img, onehot



load_training_data('dataset/normalized.bin', 751*841, 62, 55)
