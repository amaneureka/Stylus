# -*- coding: utf-8 -*-
# @Author: amaneureka
# @Date:   2017-04-07 17:41:23
# @Last Modified by:   Aman Priyadarshi
# @Last Modified time: 2017-04-17 18:05:32

import cv2
import numpy as np
import progressbar
import matplotlib.pyplot as plt


def find_samples_bounding_rect(path):

	min_w = 0
	min_h = 0

	for i in range(1, 63):
		for j in range(1, 56):

			filename = '{0}/Sample{1:03d}/img{1:03d}-{2:03d}.png'.format(path, i, j)

			# opencv read -> Gray Image -> Bounding Rect
			im = cv2.imread(filename)
			imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			imgray = cv2.bitwise_not(imgray)
			_, contours, _ = cv2.findContours(imgray, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
			_, _, w, h = cv2.boundingRect(contours[len(contours) - 1])

			# find maximum resolution
			min_w = max(min_w, w)
			min_h = max(min_h, h)

	return min_w, min_h


def crop_images(path, width, height, showsamples):

	print 'cropping images:'
	bar = progressbar.ProgressBar(maxval=62*56,
		widgets=[
		' [', progressbar.Timer(), '] ',
		progressbar.Bar(),
		' (', progressbar.ETA(), ') ',
	])
	bar.start()
	counter = 0

	with open(path + '/normalized.bin', 'wb') as f:
		img_canvas = np.zeros((height, width), dtype=np.uint8)
		for i in range(1, 63):
			for j in range(1, 56):

				filename = '{0}/Sample{1:03d}/img{1:03d}-{2:03d}.png'.format(path, i, j)

				# opencv read -> Gray Image -> Bounding Rect
				im = cv2.imread(filename)
				imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
				imgray = cv2.bitwise_not(imgray)
				_, contours, _ = cv2.findContours(imgray, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
				x, y, w, h = cv2.boundingRect(contours[len(contours) - 1])

				# center align character
				offset_x = (width - w) / 2
				offset_y = (height - h) / 2
				x = max(x - offset_x, 0)
				y = max(y - offset_y, 0)
				newimg = imgray[y : y + height, x : x + width]

				# fix images which has lower dimensions
				w = newimg.shape[1]
				h = newimg.shape[0]
				img_canvas[ : h, : w] = newimg

				# append ndarry to file
				f.write(img_canvas.flatten())

				# update progressbar
				counter = counter + 1
				bar.update(counter)

				# preview if requested
				if showsamples:
					plt.imshow(img_canvas, cmap='gray')
					plt.show()

		f.close()
	bar.finish()

if __name__ == '__main__':
	width, height = 751, 841#find_samples_bounding_rect('dataset')
	print width, height
	crop_images('dataset', width, height, False)
