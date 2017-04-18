# -*- coding: utf-8 -*-
# @Author: Aman Priyadarshi
# @Date:   2017-04-17 11:39:30
# @Last Modified by:   amaneureka
# @Last Modified time: 2017-04-19 03:57:28

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import utility
import normalize
from models import cnn

def load_training_data(filepath):

    data = np.fromfile(filepath, dtype=np.uint8)
    num_classes = int.from_bytes(data[:4], byteorder='little')
    num_samples = int.from_bytes(data[4:8], byteorder='little')
    width       = int.from_bytes(data[8:12], byteorder='little')
    height      = int.from_bytes(data[12:16], byteorder='little')

    image_flat_size = width * height

    img = data[16:]
    img.shape = (-1, image_flat_size)

    img_class = np.zeros(img.shape[0], dtype=np.int8)
    img_class.shape = (-1, num_samples)
    for i in range(num_classes):
        img_class[i, ] = i
    img_class.shape = (-1)

    # shuffle dataset
    perm = np.random.permutation(img.shape[0])
    return num_classes, width, height, img[perm], img_class[perm]


if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', action='store', dest='saver_file',
                        default='validations', help='saver file name')
    parser.add_argument('-r', action='store', dest='restore',
                        help='restore last session')
    parser.add_argument('--train', action='store', dest='train_size',
                        default=0, help='train model', type=int)
    parser.add_argument('--show', action='store_true', dest='show_data',
                        help='display loaded dataset', default=False)
    parser.add_argument('--iterations', action='store', dest='iterations',
                        help='training epochs count', type=int, default=10000)
    parser.add_argument('--batch', action='store', dest='batch_size',
                        help='training batch size', type=int, default=100)
    args = parser.parse_args()

    # load data and show data
    num_classes, width, height, img, img_class = load_training_data('dataset/normalized.bin')
    if args.show_data == True:
        # disply dataset
        fig = plt.figure()
        for i in range(1, 21):
            a = fig.add_subplot(4, 5, i)
            a.set_title('Label \'%s\'' % utility.SAMPLE(img_class[i]).name)
            plt.axis('off')
            plt.imshow(img[i].reshape(height, width), cmap='gray')
        plt.show()

    # create network and prediction parameters
    x, y, y_true, optimizer = cnn.create_network(height, width, num_classes)

    y_pred_cls = tf.argmax(y, dimension=1)
    y_true_cls = tf.placeholder(tf.int64, shape=[None])
    correct_predictions = tf.equal(y_true_cls, y_pred_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # tensorflow session saver
    saver = tf.train.Saver()
    savedir = 'saver'

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    save_path = os.path.join(savedir, args.saver_file)

    # create tensorflow session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # do we need to restore session?
    if args.restore is not None:
        saver.restore(sess=session, save_path=save_path)

    # train network of requested size
    if args.train_size > 0:

        # prepare dataset
        img_train = img[:args.train_size]
        img_class_train = img_class[:args.train_size]
        img_count = img_train.shape[0]

        img_val = img[args.train_size:]
        img_class_val = img_class[args.train_size:]
        feed_val = {x : img_val, y_true_cls : img_class_val}

        print('train data size: %d' % img_count)
        print('validation data size: %d' % img_val.shape[0])

        # convert to onehot encoding
        onehot = np.zeros((img_count, num_classes))
        onehot[np.arange(img_count), img_class_train] = 1

        best_accuracy = 0.0
        for i in range(args.iterations):
            perm = np.random.permutation(args.batch_size)
            input_x = img_train[perm]
            input_y = onehot[perm]
            feed = {x : input_x, y_true: input_y}
            session.run(optimizer, feed_dict=feed)
            # on every 100 iteration validate training
            if i % 100 == 0:
                acc = session.run(accuracy, feed_dict=feed_val)
                msg = ' '
                if acc > best_accuracy:
                    msg = '*'
                    best_accuracy = acc
                    saver.save(sess=session, save_path=save_path)
                print("validation accuracy #{0}: {2}{1:.9%}".format(i + 1, acc, msg))

