# -*- coding: utf-8 -*-
# @Author: amaneureka
# @Date:   2017-05-19 00:54:11
# @Last Modified by:   amaneureka
# @Last Modified time: 2017-05-19 12:13:45

import cv2
import pygame
import random
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import utility
from models import cnn

sample_width = 74
sample_height = 77
sample_num_class = 62

def select_color():
	return (random.randrange(256), random.randrange(256), random.randrange(256))

def roundline(srf, color, start, end, radius=1):
	dx = end[0]-start[0]
	dy = end[1]-start[1]
	distance = max(abs(dx), abs(dy))
	for i in range(distance):
		x = int(start[0]+float(i)/distance*dx)
		y = int(start[1]+float(i)/distance*dy)
		pygame.display.update(pygame.draw.circle(srf, color, (x, y), radius))

def normalize_pygame_image(screen, topleft, width, height):
	data = pygame.image.tostring(screen, 'RGB')
	img = np.fromstring(data, dtype=np.uint8)
	img.shape = (screen.get_height(), screen.get_width(), -1)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = img[topleft[1]:, topleft[0]:]
	_, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
	_, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	x, y, w, h = cv2.boundingRect(contours[len(contours) - 1])
	img = img[y:y+h, x:x+w]
	factor = min(float(width/w), float(height/h))
	img = cv2.resize(img, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)[:height, :width]
	return img

if __name__ == '__main__':

	# parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--width', action='store', dest='scr_width',
						help='screen width', type=int, default=800)
	parser.add_argument('--height', action='store', dest='scr_height',
						help='screen height', type=int, default=600)
	parser.add_argument('--radius', action='store', dest='stroke_radius',
						help='stroke radius', type=int, default=10)
	parser.add_argument('--restore', action='store', dest='restore',
						help='restore network model', required=True)
	args = parser.parse_args()

	# restore network
	x, y, _, _ = cnn.create_network(sample_height, sample_width, sample_num_class)
	y_pred_cls = tf.argmax(y, dimension=1)

	session = tf.Session()
	session.run(tf.global_variables_initializer())

	saver = tf.train.Saver()
	saver.restore(sess=session, save_path=args.restore)

	# paint program
	pygame.init()
	screen = pygame.display.set_mode((args.scr_width, args.scr_height))
	pygame.display.set_caption('Stylus')
	myfont = pygame.font.SysFont("arial", 20)
	last_pos = (0, 0)
	status_pos = (5, 10)
	enable_drawing = False
	radius = args.stroke_radius
	prev_rect = pygame.Rect(0, 0, 0, 0)
	img_canvas = np.zeros((sample_height, sample_width), dtype=np.uint8)
	try:

		while True:

			e = pygame.event.wait()
			if e.type == pygame.QUIT:
				raise StopIteration
			elif e.type == pygame.KEYDOWN:
				enable_drawing = False
				pygame.display.update(screen.fill((0, 0, 0)))
			elif e.type == pygame.MOUSEBUTTONDOWN:
				color = select_color()
				enable_drawing = True
			elif e.type == pygame.MOUSEBUTTONUP:
				enable_drawing = False
				try:
					# get bounding image
					img = normalize_pygame_image(screen, prev_rect.bottomright, sample_width, sample_height)

					w = img.shape[1]
					h = img.shape[0]
					img_canvas.fill(0)
					img_canvas[:h,:w] = img

					# plt.imshow(img_canvas, cmap='gray')
					# plt.show()

					feed = {x : img_canvas.reshape(1, -1)}
					prd_matrix, prd = session.run([y, y_pred_cls], feed_dict=feed)
					loss = prd_matrix[0, prd - 1][0]
					print(utility.SAMPLE(prd).name, loss)

					# show prediction status
					label = myfont.render("Prediction: {0} {1}".format(utility.SAMPLE(prd).name, loss), 1, (255, 255, 255))
					pygame.display.update(screen.fill((0, 0, 0), rect=prev_rect))
					pygame.display.update(screen.blit(label, status_pos))
					prev_rect = label.get_rect(topleft=status_pos)

				except:
					pass

			elif e.type == pygame.MOUSEMOTION:
				if enable_drawing:
					pygame.display.update(pygame.draw.circle(screen, color, e.pos, radius))
					roundline(screen, color, e.pos, last_pos,  radius)
				last_pos = e.pos

	except StopIteration:
		pass

	pygame.quit()
