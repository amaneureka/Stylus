# -*- coding: utf-8 -*-
# @Author: amaneureka
# @Date:   2017-04-07 17:41:23
# @Last Modified by:   amaneureka
# @Last Modified time: 2017-04-20 20:26:45

import cv2
import math
import numpy as np
import progressbar
import matplotlib.pyplot as plt

num_classes = 62
num_samples = 26

def find_samples_bounding_rect(path):

	min_w = 0
	min_h = 0

	print ('finding bounding box:')
	bar = progressbar.ProgressBar(maxval=num_classes*num_samples,
		widgets=[
		' [', progressbar.Timer(), '] ',
		progressbar.Bar(),
		' (', progressbar.ETA(), ') ',
	])
	bar.start()
	counter = 0

	for i in range(1, num_classes + 1):
		for j in range(1, num_samples + 1):

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

			# update progress bar
			counter = counter + 1
			bar.update(counter)

	bar.finish()
	return min_w, min_h


def crop_images(filename, path, samplestart, width, height, showsamples, scaling):

	print ('cropping images:')
	bar = progressbar.ProgressBar(maxval=num_classes*num_samples,
		widgets=[
		' [', progressbar.Timer(), '] ',
		progressbar.Bar(),
		' (', progressbar.ETA(), ') ',
	])
	bar.start()
	counter = 0

	new_width = int(width * scaling + 0.5)
	new_height = int(height * scaling + 0.5)

	with open(path + filename, 'wb') as f:

		# dump configs
		f.write((num_classes).to_bytes(4, byteorder='little'))
		f.write((num_samples).to_bytes(4, byteorder='little'))
		f.write((new_width).to_bytes(4, byteorder='little'))
		f.write((new_height).to_bytes(4, byteorder='little'))

		# dump images
		img_canvas = np.zeros((height, width), dtype=np.uint8)
		for i in range(1, num_classes + 1):
			for j in range(samplestart, samplestart + num_samples):

				filename = '{0}/Sample{1:03d}/img{1:03d}-{2:03d}.png'.format(path, i, j)

				# opencv read -> Gray Image -> Bounding Rect
				im = cv2.imread(filename)
				imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
				imgray = cv2.bitwise_not(imgray)
				_, contours, _ = cv2.findContours(imgray, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
				x, y, w, h = cv2.boundingRect(contours[len(contours) - 1])

				# center align character
				offset_x = int((width - w) / 2)
				offset_y = int((height - h) / 2)
				x = max(x - offset_x, 0)
				y = max(y - offset_y, 0)
				newimg = imgray[y : y + height, x : x + width]

				# fix images which has lower dimensions
				w = newimg.shape[1]
				h = newimg.shape[0]
				img_canvas[ : h, : w] = newimg

				# resize image
				newimg = cv2.resize(img_canvas, None, fx=scaling, fy=scaling, interpolation = cv2.INTER_AREA)

				# append ndarry to file
				f.write(newimg.flatten())

				# update progressbar
				counter = counter + 1
				bar.update(counter)

				# preview if requested
				if showsamples:
					plt.imshow(newimg, cmap='gray')
					plt.show()

		f.close()
	bar.finish()
	return new_width, new_height

if __name__ == '__main__':
	width, height = find_samples_bounding_rect('dataset')
	print('Bounding Rectangle:: width: %d height: %d' % (width, height))
	_, _ = crop_images('/normalized-train.bin', 'dataset', 1, width, height, False, 0.1)
	num_samples = 14
	width, height = crop_images('/normalized-val.bin', 'dataset', 14, width, height, False, 0.1)
	print('Cropping:: width: %d height: %d' % (width, height))
