#!/usr/bin/python3

#!pip3 install keras imgaug openvc-python matplotlib
#!wget https://pjreddie.com/media/files/yolov2.weights
#!git clone https://github.com/cosmicad/dataset.git
#!mkdir logs

'''
Modified from: https://github.com/experiencor/keras-yolo2
'''

import os
import sys
import cv2
import copy
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from imgaug import augmenters as iaa
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization, Flatten, Dense, Lambda
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D

image_path			= './dataset/JPEGImages/'
annot_path			= './dataset/Annotations/'
wt_path				= './yolov2.weights'
LABELS				= ['RBC']
IMAGE_H, IMAGE_W	= 416, 416
GRID_H, GRID_W		= 13, 13
BOX					= 5
CLASS				= len(LABELS)
CLASS_WEIGHTS		= np.ones(CLASS, dtype='float32')
ANCHORS				= [	0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
						5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
OBJ_THRESHOLD		= 0.3
NMS_THRESHOLD		= 0.3
NO_OBJECT_SCALE		= 1.0
OBJECT_SCALE		= 5.0
COORD_SCALE			= 1.0
CLASS_SCALE			= 1.0
BATCH_SIZE			= 16
WARM_UP_BATCHES		= 100
TRUE_BOX_BUFFER		= 50
GenConf 			= {	'IMAGE_H'			: IMAGE_H,
						'IMAGE_W'			: IMAGE_W,
						'GRID_H'			: GRID_H,
						'GRID_W'			: GRID_W,
						'BOX'				: BOX,
						'LABELS'			: LABELS,
						'CLASS'				: len(LABELS),
						'ANCHORS'			: ANCHORS,
						'BATCH_SIZE'		: BATCH_SIZE,
						'TRUE_BOX_BUFFER'	: 50,}

class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.c = c
		self.classes = classes
		self.label = -1
		self.score = -1
	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)
		return(self.label)
	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]
		return(self.score)

class WeightReader:
	def __init__(self, weight_file):
		self.offset = 4
		self.all_weights = np.fromfile(weight_file, dtype='float32')
	def read_bytes(self, size):
		self.offset = self.offset + size
		return(self.all_weights[self.offset-size:self.offset])
	def reset(self):
		self.offset = 4

def bbox_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax],\
									[box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax],\
									[box2.ymin, box2.ymax])
	intersect = intersect_w * intersect_h
	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	union = w1*h1 + w2*h2 - intersect
	return(float(intersect) / union)

def draw_boxes(image, boxes, labels):
	image_h, image_w, _ = image.shape
	for box in boxes:
		xmin = int(box.xmin*image_w)
		ymin = int(box.ymin*image_h)
		xmax = int(box.xmax*image_w)
		ymax = int(box.ymax*image_h)
		cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
		cv2.putText(image,
					labels[box.get_label()] + ' ' + str(box.get_score()),
					(xmin, ymin - 13),
					cv2.FONT_HERSHEY_SIMPLEX,
					1e-3 * image_h,
					(0,255,0), 2)
	return(image)

def decode_netout(netout, anchors, nb_class, obj_threshold=0.3,\
											nms_threshold=0.3):
	grid_h, grid_w, nb_box = netout.shape[:3]
	boxes = []
	netout[..., 4]  = _sigmoid(netout[..., 4])
	netout[..., 5:] = netout[..., 4][...,\
									np.newaxis] * _softmax(netout[..., 5:])
	netout[..., 5:] *= netout[..., 5:] > obj_threshold
	for row in range(grid_h):
		for col in range(grid_w):
			for b in range(nb_box):
				classes = netout[row,col,b,5:]
				if np.sum(classes) > 0:
					x, y, w, h = netout[row,col,b,:4]
					x = (col + _sigmoid(x)) / grid_w
					y = (row + _sigmoid(y)) / grid_h
					w = anchors[2 * b + 0] * np.exp(w) / grid_w
					h = anchors[2 * b + 1] * np.exp(h) / grid_h
					confidence = netout[row, col, b, 4]
					box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2,\
									confidence, classes)
					boxes.append(box)
	for c in range(nb_class):
		sorted_indices = list(reversed(np.argsort([box.classes[c]\
															for box in boxes])))
		for i in range(len(sorted_indices)):
			index_i = sorted_indices[i]
			if boxes[index_i].classes[c] == 0:
				continue
			else:
				for j in range(i+1, len(sorted_indices)):
					index_j = sorted_indices[j]
					if bbox_iou(boxes[index_i], boxes[index_j])>=nms_threshold:
						boxes[index_j].classes[c] = 0
	boxes = [box for box in boxes if box.get_score() > obj_threshold]
	return(boxes)

def compute_overlap(a, b):
	area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
	iw = np.minimum(np.expand_dims(a[:, 2], axis=1),\
					b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
	ih = np.minimum(np.expand_dims(a[:, 3], axis=1),\
					b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])
	iw = np.maximum(iw, 0)
	ih = np.maximum(ih, 0)
	ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]),\
														axis=1) + area - iw * ih
	ua = np.maximum(ua, np.finfo(float).eps)
	intersection = iw * ih
	return(intersection / ua)

def compute_ap(recall, precision):
	mrec = np.concatenate(([0.], recall, [1.]))
	mpre = np.concatenate(([0.], precision, [0.]))
	for i in range(mpre.size - 1, 0, -1):
		mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
	i = np.where(mrec[1:] != mrec[:-1])[0]
	ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
	return(ap)

def _interval_overlap(interval_a, interval_b):
	x1, x2 = interval_a
	x3, x4 = interval_b
	if x3 < x1:
		if x4 < x1:
			return(0)
		else:
			return(min(x2,x4) - x1)
	else:
		if x2 < x3:
				return(0)
		else:
			return(min(x2,x4) - x3)

def _sigmoid(x):
	return(1. / (1. + np.exp(-x)))

def _softmax(x, axis=-1, t=-100.):
	x = x - np.max(x)
	if np.min(x) < t:
		x = x/np.min(x)*t

def normal(image): return image/255.

def space_to_depth_x2(x): return tf.space_to_depth(x, block_size=2)

def parse_annotation(ann_dir, img_dir, labels=[]):
	all_imgs = []
	seen_labels = {}
	for ann in sorted(os.listdir(ann_dir)):
		img = {'object':[]}
		tree = ET.parse(ann_dir + ann)
		for elem in tree.iter():
			if 'filename' in elem.tag:
				img['filename'] = img_dir + elem.text
			if 'width' in elem.tag:
				img['width'] = int(elem.text)
			if 'height' in elem.tag:
				img['height'] = int(elem.text)
			if 'object' in elem.tag or 'part' in elem.tag:
				obj = {}
				for attr in list(elem):
					if 'name' in attr.tag:
						obj['name'] = attr.text
						if obj['name'] in seen_labels:
							seen_labels[obj['name']] += 1
						else:
							seen_labels[obj['name']] = 1
						if len(labels) > 0 and obj['name'] not in labels:
							break
						else:
							img['object'] += [obj]
					if 'bndbox' in attr.tag:
						for dim in list(attr):
							if 'xmin' in dim.tag:
								obj['xmin'] = int(round(float(dim.text)))
							if 'ymin' in dim.tag:
								obj['ymin'] = int(round(float(dim.text)))
							if 'xmax' in dim.tag:
								obj['xmax'] = int(round(float(dim.text)))
							if 'ymax' in dim.tag:
								obj['ymax'] = int(round(float(dim.text)))
		if len(img['object']) > 0:
			all_imgs += [img]
	return(all_imgs, seen_labels)

class BatchGenerator(keras.utils.Sequence):
	def __init__(self, images, config, shuffle=True, jitter=True, norm=None):
		self.generator = None
		self.images = images
		self.config = config
		self.shuffle = shuffle
		self.jitter = jitter
		self.norm = norm
		self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i],\
						config['ANCHORS'][2*i+1]) for i in range(int(len(\
						config['ANCHORS'])//2))]
		sometimes = lambda aug: iaa.Sometimes(0.5, aug)
		self.aug_pipe = iaa.Sequential(
			[
				sometimes(iaa.Affine(
				)),
				iaa.SomeOf((0, 5),
					[
						iaa.OneOf([
							iaa.GaussianBlur((0, 3.0)),
							iaa.AverageBlur(k=(2, 7)),
							iaa.MedianBlur(k=(3, 11)),
						]),
						iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
						iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255),\
													per_channel=0.5),
						iaa.OneOf([
							iaa.Dropout((0.01, 0.1), per_channel=0.5),
						]),
						iaa.Add((-10, 10), per_channel=0.5),
						iaa.Multiply((0.5, 1.5), per_channel=0.5),
						iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
					],
					random_order=True
				)
			],
			random_order=True
		)
		if shuffle: np.random.shuffle(self.images)
	def __len__(self):
		return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))
	def num_classes(self):
		return len(self.config['LABELS'])
	def size(self):
		return len(self.images)
	def load_annotation(self, i):
		annots = []
		for obj in self.images[i]['object']:
			annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'],\
					self.config['LABELS'].index(obj['name'])]
			annots += [annot]
		if len(annots) == 0: annots = [[]]
		return np.array(annots)
	def load_image(self, i):
		return cv2.imread(self.images[i]['filename'])
	def __getitem__(self, idx):
		l_bound = idx*self.config['BATCH_SIZE']
		r_bound = (idx+1)*self.config['BATCH_SIZE']
		if r_bound > len(self.images):
			r_bound = len(self.images)
			l_bound = r_bound - self.config['BATCH_SIZE']
		instance_count = 0
		x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'],\
							self.config['IMAGE_W'], 3))
		b_batch = np.zeros((r_bound - l_bound, 1, 1, 1,\
							self.config['TRUE_BOX_BUFFER'], 4))
		y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],\
							self.config['GRID_W'], self.config['BOX'],\
							4+1+len(self.config['LABELS'])))
		for train_instance in self.images[l_bound:r_bound]:
			img, all_objs = self.aug_image(train_instance, jitter=self.jitter)
			true_box_index = 0
			for obj in all_objs:
				if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and\
										obj['name'] in self.config['LABELS']:
					center_x = .5*(obj['xmin'] + obj['xmax'])
					center_x = center_x / (float(self.config['IMAGE_W']) /\
										self.config['GRID_W'])
					center_y = .5*(obj['ymin'] + obj['ymax'])
					center_y = center_y / (float(self.config['IMAGE_H']) /\
										self.config['GRID_H'])
					grid_x = int(np.floor(center_x))
					grid_y = int(np.floor(center_y))
					if grid_x < self.config['GRID_W'] and grid_y <\
										self.config['GRID_H']:
						obj_indx  = self.config['LABELS'].index(obj['name'])
						center_w = (obj['xmax'] - obj['xmin']) /\
										(float(self.config['IMAGE_W']) /\
										self.config['GRID_W'])
						center_h = (obj['ymax'] - obj['ymin']) /\
										(float(self.config['IMAGE_H']) /\
										self.config['GRID_H'])
						box = [center_x, center_y, center_w, center_h]
						best_anchor = -1
						max_iou     = -1
						shifted_box = BoundBox(	0,
												0,
												center_w,
												center_h)
						for i in range(len(self.anchors)):
							anchor= self.anchors[i]
							iou = bbox_iou(shifted_box, anchor)
							if max_iou < iou:
								best_anchor = i
								max_iou     = iou
						y_batch[instance_count, grid_y, grid_x,\
													best_anchor, 0:4] = box
						y_batch[instance_count, grid_y, grid_x, best_anchor,\
													4] = 1.
						y_batch[instance_count, grid_y, grid_x, best_anchor,\
													5+obj_indx] = 1
						b_batch[instance_count, 0, 0, 0, true_box_index] = box
						true_box_index += 1
						true_box_index = true_box_index\
										% self.config['TRUE_BOX_BUFFER']
			if self.norm != None:
				x_batch[instance_count] = self.norm(img)
			else:
				for obj in all_objs:
					if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
						cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']),\
									(obj['xmax'],obj['ymax']), (255,0,0), 3)
						cv2.putText(img[:,:,::-1], obj['name'],
									(obj['xmin']+2, obj['ymin']+12),
									0, 1.2e-3 * img.shape[0],
									(0,255,0), 2)
				x_batch[instance_count] = img
			instance_count += 1
		return [x_batch, b_batch], y_batch
	def on_epoch_end(self):
		if self.shuffle: np.random.shuffle(self.images)
	def aug_image(self, train_instance, jitter):
		image_name = train_instance['filename']
		image = cv2.imread(image_name)
		if image is None: print('Cannot find ', image_name)
		h, w, c = image.shape
		all_objs = copy.deepcopy(train_instance['object'])
		if jitter:
			scale = np.random.uniform() / 10. + 1.
			image = cv2.resize(image, (0,0), fx = scale, fy = scale)
			max_offx = (scale-1.) * w
			max_offy = (scale-1.) * h
			offx = int(np.random.uniform() * max_offx)
			offy = int(np.random.uniform() * max_offy)
			image = image[offy : (offy + h), offx : (offx + w)]
			flip = np.random.binomial(1, .5)
			if flip > 0.5: image = cv2.flip(image, 1)
			image = self.aug_pipe.augment_image(image)
		image = cv2.resize(image, (self.config['IMAGE_H'],\
									self.config['IMAGE_W']))
		image = image[:,:,::-1]
		for obj in all_objs:
			for attr in ['xmin', 'xmax']:
				if jitter: obj[attr] = int(obj[attr] * scale - offx)
				obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
				obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)
			for attr in ['ymin', 'ymax']:
				if jitter: obj[attr] = int(obj[attr] * scale - offy)
				obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
				obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)
			if jitter and flip > 0.5:
				xmin = obj['xmin']
				obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
				obj['xmax'] = self.config['IMAGE_W'] - xmin
		return(image, all_objs)

all_imgs, seen_labels = parse_annotation(annot_path, image_path)
for img in all_imgs: img['filename'] = img['filename'] + '.jpg'
batches = BatchGenerator(all_imgs, GenConf)
image = batches[0][0][0][0]
train_valid_split = int(0.8*len(all_imgs))
train_batch = BatchGenerator(all_imgs[:train_valid_split], GenConf)
valid_batch = BatchGenerator(all_imgs[train_valid_split:], GenConf, norm=normal)
input_image = keras.layers.Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes  = keras.layers.Input(shape=(1, 1, 1, TRUE_BOX_BUFFER, 4))

x = Conv2D(32,(3,3),strides=(1,1),padding='same',name='conv_1',use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(64,(3,3),strides=(1,1),padding='same',name='conv_2',use_bias=False)(x)
x = BatchNormalization(name='norm_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(128,(3,3),strides=(1,1),padding='same',name='conv_3',use_bias=False)(x)
x = BatchNormalization(name='norm_3')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(64,(1,1),strides=(1,1),padding='same',name='conv_4',use_bias=False)(x)
x = BatchNormalization(name='norm_4')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(128,(3,3),strides=(1,1),padding='same',name='conv_5',use_bias=False)(x)
x = BatchNormalization(name='norm_5')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(256,(3,3),strides=(1,1),padding='same',name='conv_6',use_bias=False)(x)
x = BatchNormalization(name='norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(128,(1,1),strides=(1,1),padding='same',name='conv_7', use_bias=False)(x)
x = BatchNormalization(name='norm_7')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(256,(3,3),strides=(1,1),padding='same',name='conv_8',use_bias=False,input_shape=(416,416,3))(x)
x = BatchNormalization(name='norm_8')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(512,(3,3),strides=(1,1),padding='same',name='conv_9',use_bias=False)(x)
x = BatchNormalization(name='norm_9')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(256,(1,1),strides=(1,1),padding='same',name='conv_10',use_bias=False)(x)
x = BatchNormalization(name='norm_10')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(512,(3,3),strides=(1,1),padding='same',name='conv_11',use_bias=False)(x)
x = BatchNormalization(name='norm_11')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(256,(1,1),strides=(1,1),padding='same',name='conv_12',use_bias=False)(x)
x = BatchNormalization(name='norm_12')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(512,(3,3),strides=(1,1),padding='same',name='conv_13',use_bias=False)(x)
x = BatchNormalization(name='norm_13')(x)
x = LeakyReLU(alpha=0.1)(x)
skip_connection = x
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv_14',use_bias=False)(x)
x = BatchNormalization(name='norm_14')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(512,(1,1),strides=(1,1),padding='same',name='conv_15',use_bias=False)(x)
x = BatchNormalization(name='norm_15')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv_16',use_bias=False)(x)
x = BatchNormalization(name='norm_16')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(512,(1,1),strides=(1,1),padding='same',name='conv_17',use_bias=False)(x)
x = BatchNormalization(name='norm_17')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv_18',use_bias=False)(x)
x = BatchNormalization(name='norm_18')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv_19',use_bias=False)(x)
x = BatchNormalization(name='norm_19')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv_20',use_bias=False)(x)
x = BatchNormalization(name='norm_20')(x)
x = LeakyReLU(alpha=0.1)(x)
skip_connection = Conv2D(64,(1,1),strides=(1,1),padding='same',name='conv_21',use_bias=False)(skip_connection)
skip_connection = BatchNormalization(name='norm_21')(skip_connection)
skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
skip_connection = Lambda(space_to_depth_x2)(skip_connection)
x = concatenate([skip_connection, x])
x = Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv_22',use_bias=False)(x)
x = BatchNormalization(name='norm_22')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(BOX*(4+1+CLASS),(1,1),strides=(1,1),padding='same',name='conv_23')(x)
output = Reshape((GRID_H,GRID_W,BOX,4+1+CLASS))(x)
output = Lambda(lambda args: args[0])([output,true_boxes])
model = keras.models.Model([input_image,true_boxes],output)
#model.summary()

weight_reader = WeightReader(wt_path)
weight_reader.reset()
nb_conv = 23
for i in range(1, nb_conv+1):
	conv_layer = model.get_layer('conv_' + str(i))
	if i < nb_conv:
		norm_layer = model.get_layer('norm_' + str(i))
		size = np.prod(norm_layer.get_weights()[0].shape)
		beta = weight_reader.read_bytes(size)
		gamma = weight_reader.read_bytes(size)
		mean = weight_reader.read_bytes(size)
		var = weight_reader.read_bytes(size)
		weights = norm_layer.set_weights([gamma, beta, mean, var])
	if len(conv_layer.get_weights()) > 1:
		bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()\
																	[1].shape))
		kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()\
																	[0].shape))
		kernel = kernel.reshape(list(reversed(conv_layer.get_weights()\
																	[0].shape)))
		kernel = kernel.transpose([2,3,1,0])
		conv_layer.set_weights([kernel, bias])
	else:
		kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()\
																	[0].shape))
		kernel = kernel.reshape(list(reversed(conv_layer.get_weights()\
																	[0].shape)))
		kernel = kernel.transpose([2,3,1,0])
		conv_layer.set_weights([kernel])
layer		= model.layers[-4]
weights		= layer.get_weights()
new_kernel	= np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
new_bias	= np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)
layer.set_weights([new_kernel, new_bias])

def custom_loss(y_true, y_pred):
	mask_shape		= tf.shape(y_true)[:4]
	cell_x			= tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W),\
										[GRID_H]), (1, GRID_H, GRID_W, 1, 1)))
	cell_y			= tf.transpose(cell_x, (0,2,1,3,4))
	cell_grid		= tf.tile(tf.concat([cell_x,cell_y], -1),\
										[BATCH_SIZE, 1, 1, 5, 1])
	coord_mask		= tf.zeros(mask_shape)
	conf_mask		= tf.zeros(mask_shape)
	class_mask		= tf.zeros(mask_shape)
	seen			= tf.Variable(0.)
	total_recall	= tf.Variable(0.)
	pred_box_xy		= tf.sigmoid(y_pred[..., :2]) + cell_grid
	pred_box_wh		= tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS,\
										[1,1,1,BOX,2])
	pred_box_conf	= tf.sigmoid(y_pred[..., 4])
	pred_box_class	= y_pred[..., 5:]
	true_box_xy		= y_true[..., 0:2]
	true_box_wh		= y_true[..., 2:4]
	true_wh_half	= true_box_wh / 2.
	true_mins		= true_box_xy - true_wh_half
	true_maxes		= true_box_xy + true_wh_half
	pred_wh_half	= pred_box_wh / 2.
	pred_mins		= pred_box_xy - pred_wh_half
	pred_maxes		= pred_box_xy + pred_wh_half
	intersect_mins	= tf.maximum(pred_mins,  true_mins)
	intersect_maxes	= tf.minimum(pred_maxes, true_maxes)
	intersect_wh	= tf.maximum(intersect_maxes - intersect_mins, 0.)
	intersect_areas	= intersect_wh[..., 0] * intersect_wh[..., 1]
	true_areas		= true_box_wh[..., 0] * true_box_wh[..., 1]
	pred_areas		= pred_box_wh[..., 0] * pred_box_wh[..., 1]
	union_areas		= pred_areas + true_areas - intersect_areas
	iou_scores		= tf.truediv(intersect_areas, union_areas)
	true_box_conf	= iou_scores * y_true[..., 4]
	true_box_class	= tf.argmax(y_true[..., 5:], -1)
	coord_mask		= tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE
	true_xy			= true_boxes[..., 0:2]
	true_wh			= true_boxes[..., 2:4]
	true_wh_half	= true_wh / 2.
	true_mins		= true_xy - true_wh_half
	true_maxes		= true_xy + true_wh_half
	pred_xy			= tf.expand_dims(pred_box_xy, 4)
	pred_wh			= tf.expand_dims(pred_box_wh, 4)
	pred_wh_half	= pred_wh / 2.
	pred_mins		= pred_xy - pred_wh_half
	pred_maxes		= pred_xy + pred_wh_half
	intersect_mins	= tf.maximum(pred_mins,  true_mins)
	intersect_maxes	= tf.minimum(pred_maxes, true_maxes)
	intersect_wh	= tf.maximum(intersect_maxes - intersect_mins, 0.)
	intersect_areas	= intersect_wh[..., 0] * intersect_wh[..., 1]
	true_areas		= true_wh[..., 0] * true_wh[..., 1]
	pred_areas		= pred_wh[..., 0] * pred_wh[..., 1]
	union_areas		= pred_areas + true_areas - intersect_areas
	iou_scores		= tf.truediv(intersect_areas, union_areas)
	best_ious		= tf.reduce_max(iou_scores, axis=4)
	conf_mask		= conf_mask + tf.to_float(best_ious < 0.6) *\
										(1 - y_true[..., 4]) * NO_OBJECT_SCALE
	conf_mask		= conf_mask + y_true[..., 4] * OBJECT_SCALE
	class_mask		= y_true[..., 4] * tf.gather(CLASS_WEIGHTS,\
												true_box_class) * CLASS_SCALE
	no_boxes_mask	= tf.to_float(coord_mask < COORD_SCALE/2.)
	seen			= tf.assign_add(seen, 1.)
	true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen,\
															WARM_UP_BATCHES),
							lambda: [true_box_xy + (0.5 + cell_grid) *\
									no_boxes_mask, true_box_wh +\
									tf.ones_like(true_box_wh) *\
									np.reshape(ANCHORS, [1,1,1,BOX,2]) *\
									no_boxes_mask, tf.ones_like(coord_mask)],
							lambda: [true_box_xy, true_box_wh, coord_mask])
	nb_coord_box	= tf.reduce_sum(tf.to_float(coord_mask > 0.0))
	nb_conf_box		= tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
	nb_class_box	= tf.reduce_sum(tf.to_float(class_mask > 0.0))
	loss_xy			= tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)\
								* coord_mask)	/	(nb_coord_box + 1e-6) / 2.
	loss_wh			= tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)\
								* coord_mask)	/	(nb_coord_box + 1e-6) / 2.
	loss_conf		= tf.reduce_sum(tf.square(true_box_conf-pred_box_conf)\
								* conf_mask)	/	(nb_conf_box  + 1e-6) / 2.
	loss_class		= tf.nn.sparse_softmax_cross_entropy_with_logits\
								(labels=true_box_class, logits=pred_box_class)
	loss_class		= tf.reduce_sum(loss_class * class_mask) /\
								(nb_class_box + 1e-6)
	loss = loss_xy + loss_wh + loss_conf + loss_class
	nb_true_box = tf.reduce_sum(y_true[..., 4])
	nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) *\
											tf.to_float(pred_box_conf > 0.3))
	current_recall = nb_pred_box/(nb_true_box + 1e-6)
	total_recall = tf.assign_add(total_recall, current_recall)
	loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t',\
																summarize=1000)
	loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
	loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
	loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
	loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
	loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
	loss = tf.Print(loss, [current_recall], message='Current Recall \t',\
																summarize=1000)
	loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t',\
																summarize=1000)
	return(loss)

early_stop = keras.callbacks.EarlyStopping(		monitor='val_loss',
												min_delta=0.001,
												patience=3,
												mode='min',
												verbose=0)
checkpoint = keras.callbacks.ModelCheckpoint(	'weights.h5',
												monitor='val_loss',
												verbose=0,
												save_best_only=True,
												mode='min',
												period=1)

tb_counter = len([log for log in os.listdir(os.path.expanduser('./logs/'))\
														if 'cell' in log]) + 1
tensorboard = keras.callbacks.TensorBoard(	log_dir=os.path.expanduser\
											('./logs/') +\
											'cell' + '_' + str(tb_counter),
											histogram_freq=0,
											write_graph=True,
											write_images=False)

optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
#optimizer = RMSprop(lr=1e-5, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss=custom_loss, optimizer=optimizer)

if sys.argv[1] == 'train':
	model.fit_generator(generator		= train_batch,
						steps_per_epoch	= len(train_batch),
						epochs			= 100,
						verbose			= 1,
						validation_data	= valid_batch,
						validation_steps= len(valid_batch),
						callbacks		= [early_stop, checkpoint, tensorboard],
						max_queue_size	= 3)

elif sys.argv[1] == 'detect':
	model.load_weights('weights.h5')
	dummy_array = np.zeros((1, 1, 1, 1, TRUE_BOX_BUFFER, 4))
	image = cv2.imread(sys.argv[2])
	plt.figure(figsize=(10,10))
	input_image = cv2.resize(image, (416, 416))
	input_image = input_image / 255.
	input_image = input_image[:,:,::-1]
	input_image = np.expand_dims(input_image, 0)
	netout = model.predict([input_image, dummy_array])
	boxes = decode_netout(netout[0],
							obj_threshold=0.5,
							nms_threshold=NMS_THRESHOLD,
							anchors=ANCHORS,
							nb_class=CLASS)
	image = draw_boxes(image, boxes, labels=LABELS)
	plt.imshow(image[:,:,::-1]); plt.show()

else:
	pass
