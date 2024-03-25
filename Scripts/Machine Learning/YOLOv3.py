#!/usr/bin/env python3

'''
MIT License

Copyright (c) 2017 Ngoc Anh Huynh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This script is modified from https://github.com/experiencor/keras-yolo3
'''

import os
import sys
import cv2
import copy
import json
import scipy
import keras
import pickle
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from keras.utils import Sequence
from keras.optimizers import Adam
from keras.engine.topology import Layer
from keras.models import Model, load_model
from keras.layers.merge import add, concatenate
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Lambda, concatenate, ZeroPadding2D, UpSampling2D, Lambda, Conv2D, Input, BatchNormalization, LeakyReLU

config = {"model":{
			"min_input_size":       288,
			"max_input_size":       448,
			"anchors":              [55,69,75,234,133,240,136,129,142,363,203,290,228,184,285,359,341,260],
			"labels":               ["Cell"]},
		"train":{
			"train_image_folder":   "./dataset/Images/",
			"train_annot_folder":   "./dataset/Annotations/",
			"tensorboard_dir":      "./logs",
			"saved_weights_name":   "cell.h5",
			"cache_name":           "cell.plk",
			"pretrained_weights":   "",
			"train_times":          8,
			"batch_size":           16,
			"learning_rate":        1e-4,
			"nb_epochs":            1000,
			"warmup_epochs":        0,
			"ignore_thresh":        0.5,
			"gpus":                 "0,1",
			"grid_scales":          [1,1,1],
			"obj_scale":            5,
			"noobj_scale":          1,
			"xywh_scale":           1,
			"class_scale":          1,
			"debug":                True},
		"valid":{
			"valid_image_folder":   "",
			"valid_annot_folder":   "",
			"cache_name":           "",
			"valid_times":          1}}

class YoloLayer(Layer):
	def __init__(self, anchors, max_grid, batch_size, warmup_batches, ignore_thresh, grid_scale, obj_scale, noobj_scale, xywh_scale, class_scale, **kwargs):
		# make the model settings persistent
		self.ignore_thresh  = ignore_thresh
		self.warmup_batches = warmup_batches
		self.anchors        = tf.constant(anchors, dtype='float', shape=[1,1,1,3,2])
		self.grid_scale     = grid_scale
		self.obj_scale      = obj_scale
		self.noobj_scale    = noobj_scale
		self.xywh_scale     = xywh_scale
		self.class_scale    = class_scale
		# make a persistent mesh grid
		max_grid_h, max_grid_w = max_grid
		cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
		cell_y = tf.transpose(cell_x, (0,2,1,3,4))
		self.cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [batch_size, 1, 1, 3, 1])
		super(YoloLayer, self).__init__(**kwargs)
	def build(self, input_shape):
		super(YoloLayer, self).build(input_shape) # Be sure to call this somewhere!
	def call(self, x):
		input_image, y_pred, y_true, true_boxes = x
		# adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
		y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))
		# initialize the masks
		object_mask     = tf.expand_dims(y_true[..., 4], 4)
		# the variable to keep track of number of batches processed
		batch_seen = tf.Variable(0.)
		# compute grid factor and net factor
		grid_h          = tf.shape(y_true)[1]
		grid_w          = tf.shape(y_true)[2]
		grid_factor     = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])
		net_h           = tf.shape(input_image)[1]
		net_w           = tf.shape(input_image)[2]
		net_factor      = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1,1,1,1,2])
		'''
		Adjust prediction
		'''
		pred_box_xy     = (self.cell_grid[:,:grid_h,:grid_w,:,:] + tf.sigmoid(y_pred[..., :2])) # sigma(t_xy) + c_xy
		pred_box_wh     = y_pred[..., 2:4] # t_wh
		pred_box_conf   = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4) # adjust confidence
		pred_box_class  = y_pred[..., 5:] # adjust class probabilities
		'''
		Adjust ground truth
		'''
		true_box_xy     = y_true[..., 0:2] # (sigma(t_xy) + c_xy)
		true_box_wh     = y_true[..., 2:4] # t_wh
		true_box_conf   = tf.expand_dims(y_true[..., 4], 4)
		true_box_class  = tf.argmax(y_true[..., 5:], -1)
		'''
		Compare each predicted box to all true boxes
		'''
		# initially, drag all objectness of all boxes to 0
		conf_delta      = pred_box_conf - 0
		# then, ignore the boxes which have good overlap with some true box
		true_xy         = true_boxes[..., 0:2] / grid_factor
		true_wh         = true_boxes[..., 2:4] / net_factor
		true_wh_half    = true_wh / 2.
		true_mins       = true_xy - true_wh_half
		true_maxes      = true_xy + true_wh_half
		pred_xy         = tf.expand_dims(pred_box_xy / grid_factor, 4)
		pred_wh         = tf.expand_dims(tf.exp(pred_box_wh) * self.anchors / net_factor, 4)
		pred_wh_half    = pred_wh / 2.
		pred_mins       = pred_xy - pred_wh_half
		pred_maxes      = pred_xy + pred_wh_half
		intersect_mins  = tf.maximum(pred_mins,  true_mins)
		intersect_maxes = tf.minimum(pred_maxes, true_maxes)
		intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
		intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
		true_areas      = true_wh[..., 0] * true_wh[..., 1]
		pred_areas      = pred_wh[..., 0] * pred_wh[..., 1]
		union_areas     = pred_areas + true_areas - intersect_areas
		iou_scores      = tf.truediv(intersect_areas, union_areas)
		best_ious       = tf.reduce_max(iou_scores, axis=4)
		conf_delta     *= tf.expand_dims(tf.to_float(best_ious < self.ignore_thresh), 4)
		'''
		Compute some online statistics
		'''
		true_xy         = true_box_xy / grid_factor
		true_wh         = tf.exp(true_box_wh) * self.anchors / net_factor
		true_wh_half    = true_wh / 2.
		true_mins       = true_xy - true_wh_half
		true_maxes      = true_xy + true_wh_half
		pred_xy         = pred_box_xy / grid_factor
		pred_wh         = tf.exp(pred_box_wh) * self.anchors / net_factor
		pred_wh_half    = pred_wh / 2.
		pred_mins       = pred_xy - pred_wh_half
		pred_maxes      = pred_xy + pred_wh_half
		intersect_mins  = tf.maximum(pred_mins,  true_mins)
		intersect_maxes = tf.minimum(pred_maxes, true_maxes)
		intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
		intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
		true_areas      = true_wh[..., 0] * true_wh[..., 1]
		pred_areas      = pred_wh[..., 0] * pred_wh[..., 1]
		union_areas     = pred_areas + true_areas - intersect_areas
		iou_scores      = tf.truediv(intersect_areas, union_areas)
		iou_scores      = object_mask * tf.expand_dims(iou_scores, 4)
		count           = tf.reduce_sum(object_mask)
		count_noobj     = tf.reduce_sum(1 - object_mask)
		detect_mask     = tf.to_float((pred_box_conf*object_mask) >= 0.5)
		class_mask      = tf.expand_dims(tf.to_float(tf.equal(tf.argmax(pred_box_class, -1), true_box_class)), 4)
		recall50        = tf.reduce_sum(tf.to_float(iou_scores >= 0.5 ) * detect_mask  * class_mask) / (count + 1e-3)
		recall75        = tf.reduce_sum(tf.to_float(iou_scores >= 0.75) * detect_mask  * class_mask) / (count + 1e-3)
		avg_iou         = tf.reduce_sum(iou_scores) / (count + 1e-3)
		avg_obj         = tf.reduce_sum(pred_box_conf  * object_mask)  / (count + 1e-3)
		avg_noobj       = tf.reduce_sum(pred_box_conf  * (1-object_mask))  / (count_noobj + 1e-3)
		avg_cat         = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3)
		'''
		Warm-up training
		'''
		batch_seen      = tf.assign_add(batch_seen, 1.)
		true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches+1), lambda: [true_box_xy + (0.5 + self.cell_grid[:,:grid_h,:grid_w,:,:]) * (1-object_mask), true_box_wh + tf.zeros_like(true_box_wh) * (1-object_mask), tf.ones_like(object_mask)], lambda: [true_box_xy, true_box_wh, object_mask])
		'''
		Compare each true box to all anchor boxes
		'''
		wh_scale        = tf.exp(true_box_wh) * self.anchors / net_factor
		wh_scale        = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) # the smaller the box, the bigger the scale
		xy_delta        = xywh_mask     * (pred_box_xy-true_box_xy) * wh_scale * self.xywh_scale
		wh_delta        = xywh_mask     * (pred_box_wh-true_box_wh) * wh_scale * self.xywh_scale
		conf_delta      = object_mask   * (pred_box_conf-true_box_conf) * self.obj_scale + (1-object_mask) * conf_delta * self.noobj_scale
		class_delta     = object_mask   * tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class), 4) * self.class_scale
		loss_xy         = tf.reduce_sum(tf.square(xy_delta),        list(range(1,5)))
		loss_wh         = tf.reduce_sum(tf.square(wh_delta),        list(range(1,5)))
		loss_conf       = tf.reduce_sum(tf.square(conf_delta),      list(range(1,5)))
		loss_class      = tf.reduce_sum(class_delta,                list(range(1,5)))
		loss            = loss_xy + loss_wh + loss_conf + loss_class
		loss            = tf.Print(loss, [grid_h, avg_obj], message='avg_obj \t\t', summarize=1000)
		loss            = tf.Print(loss, [grid_h, avg_noobj], message='avg_noobj \t\t', summarize=1000)
		loss            = tf.Print(loss, [grid_h, avg_iou], message='avg_iou \t\t', summarize=1000)
		loss            = tf.Print(loss, [grid_h, avg_cat], message='avg_cat \t\t', summarize=1000)
		loss            = tf.Print(loss, [grid_h, recall50], message='recall50 \t', summarize=1000)
		loss            = tf.Print(loss, [grid_h, recall75], message='recall75 \t', summarize=1000)
		loss            = tf.Print(loss, [grid_h, count], message='count \t', summarize=1000)
		loss            = tf.Print(loss, [grid_h, tf.reduce_sum(loss_xy), tf.reduce_sum(loss_wh), tf.reduce_sum(loss_conf), tf.reduce_sum(loss_class)], message='loss xy, wh, conf, class: \t', summarize=1000)
		return loss*self.grid_scale
	def compute_output_shape(self, input_shape):
		return [(None, 1)]

def _rand_scale(scale):
	scale = np.random.uniform(1, scale)
	return scale if (np.random.randint(2) == 0) else 1./scale;

def _constrain(min_v, max_v, value):
	if value < min_v: return min_v
	if value > max_v: return max_v
	return value

def random_flip(image, flip):
	if flip == 1: return cv2.flip(image, 1)
	return image

def correct_bounding_boxes(boxes, new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h):
	boxes = copy.deepcopy(boxes)
	# randomize boxes' order
	np.random.shuffle(boxes)
	# correct sizes and positions
	sx, sy = float(new_w)/image_w, float(new_h)/image_h
	zero_boxes = []
	for i in range(len(boxes)):
		boxes[i]['xmin'] = int(_constrain(0, net_w, boxes[i]['xmin']*sx + dx))
		boxes[i]['xmax'] = int(_constrain(0, net_w, boxes[i]['xmax']*sx + dx))
		boxes[i]['ymin'] = int(_constrain(0, net_h, boxes[i]['ymin']*sy + dy))
		boxes[i]['ymax'] = int(_constrain(0, net_h, boxes[i]['ymax']*sy + dy))
		if boxes[i]['xmax'] <= boxes[i]['xmin'] or boxes[i]['ymax'] <= boxes[i]['ymin']:
			zero_boxes += [i]
			continue
		if flip == 1:
			swap = boxes[i]['xmin'];
			boxes[i]['xmin'] = net_w - boxes[i]['xmax']
			boxes[i]['xmax'] = net_w - swap
	boxes = [boxes[i] for i in range(len(boxes)) if i not in zero_boxes]
	return boxes

def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
	# determine scale factors
	dhue = np.random.uniform(-hue, hue)
	dsat = _rand_scale(saturation);
	dexp = _rand_scale(exposure);
	# convert RGB space to HSV space
	image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')
	# change satuation and exposure
	image[:,:,1] *= dsat
	image[:,:,2] *= dexp
	# change hue
	image[:,:,0] += dhue
	image[:,:,0] -= (image[:,:,0] > 180)*180
	image[:,:,0] += (image[:,:,0] < 0)  *180
	# convert back to RGB from HSV
	return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)

def apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy):
	im_sized = cv2.resize(image, (new_w, new_h))
	if dx > 0:
		im_sized = np.pad(im_sized, ((0,0), (dx,0), (0,0)), mode='constant', constant_values=127)
	else:
		im_sized = im_sized[:,-dx:,:]
	if (new_w + dx) < net_w:
		im_sized = np.pad(im_sized, ((0,0), (0, net_w - (new_w+dx)), (0,0)), mode='constant', constant_values=127)
	if dy > 0:
		im_sized = np.pad(im_sized, ((dy,0), (0,0), (0,0)), mode='constant', constant_values=127)
	else:
		im_sized = im_sized[-dy:,:,:]
	if (new_h + dy) < net_h:
		im_sized = np.pad(im_sized, ((0, net_h - (new_h+dy)), (0,0), (0,0)), mode='constant', constant_values=127)
	return im_sized[:net_h, :net_w,:]

def _conv_block(inp, convs, do_skip=True):
	x = inp
	count = 0
	for conv in convs:
		if count == (len(convs) - 2) and do_skip:
			skip_connection = x
		count += 1
		if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # unlike tensorflow darknet prefer left and top paddings
		x = Conv2D(conv['filter'],
					conv['kernel'],
					strides=conv['stride'],
					padding='valid' if conv['stride'] > 1 else 'same', # unlike tensorflow darknet prefer left and top paddings
					name='conv_' + str(conv['layer_idx']),
					use_bias=False if conv['bnorm'] else True)(x)
		if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
		if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)
	return add([skip_connection, x]) if do_skip else x

def create_yolov3_model(nb_class, anchors, max_box_per_image, max_grid, batch_size, warmup_batches, ignore_thresh, grid_scales, obj_scale, noobj_scale, xywh_scale, class_scale):
	input_image     = Input(shape=(None, None, 3)) # net_h, net_w, 3
	true_boxes      = Input(shape=(1, 1, 1, max_box_per_image, 4))
	true_yolo_1     = Input(shape=(None, None, len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class
	true_yolo_2     = Input(shape=(None, None, len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class
	true_yolo_3     = Input(shape=(None, None, len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class
	# Layer  0 => 4
	x = _conv_block(input_image, [	{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
									{'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
									{'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
									{'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])
	# Layer  5 => 8
	x = _conv_block(x, [			{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
									{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
									{'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])
	# Layer  9 => 11
	x = _conv_block(x, [			{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
									{'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])
	# Layer 12 => 15
	x = _conv_block(x, [			{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
									{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
									{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])
	# Layer 16 => 36
	for i in range(7):
		x = _conv_block(x, [		{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
									{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])
	skip_36 = x
	# Layer 37 => 40
	x = _conv_block(x, [			{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
									{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
									{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])
	# Layer 41 => 61
	for i in range(7):
		x = _conv_block(x, [		{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
									{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])
	skip_61 = x
	# Layer 62 => 65
	x = _conv_block(x, [			{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
									{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
									{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])
	# Layer 66 => 74
	for i in range(3):
		x = _conv_block(x, [		{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
									{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])
	# Layer 75 => 79
	x = _conv_block(x, [			{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
									{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
									{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
									{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
									{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], do_skip=False)
	# Layer 80 => 82
	pred_yolo_1 = _conv_block(x, [	{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
									{'filter': (3*(5+nb_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], do_skip=False)
	loss_yolo_1 = YoloLayer(anchors[12:], [1*num for num in max_grid], batch_size, warmup_batches, ignore_thresh, grid_scales[0], obj_scale, noobj_scale, xywh_scale, class_scale)([input_image, pred_yolo_1, true_yolo_1, true_boxes])
	# Layer 83 => 86
	x = _conv_block(x, [			{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], do_skip=False)
	x = UpSampling2D(2)(x)
	x = concatenate([x, skip_61])
	# Layer 87 => 91
	x = _conv_block(x, [			{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
									{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
									{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
									{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
									{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], do_skip=False)
	# Layer 92 => 94
	pred_yolo_2 = _conv_block(x, [	{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
									{'filter': (3*(5+nb_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}], do_skip=False)
	loss_yolo_2 = YoloLayer(anchors[6:12], [2*num for num in max_grid], batch_size, warmup_batches, ignore_thresh, grid_scales[1], obj_scale, noobj_scale, xywh_scale, class_scale)([input_image, pred_yolo_2, true_yolo_2, true_boxes])
	# Layer 95 => 98
	x = _conv_block(x, [			{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], do_skip=False)
	x = UpSampling2D(2)(x)
	x = concatenate([x, skip_36])
	# Layer 99 => 106
	pred_yolo_3 = _conv_block(x, [	{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
									{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
									{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
									{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
									{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
									{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},
									{'filter': (3*(5+nb_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}], do_skip=False)
	loss_yolo_3 = YoloLayer(anchors[:6], [4*num for num in max_grid], batch_size, warmup_batches, ignore_thresh, grid_scales[2], obj_scale, noobj_scale, xywh_scale, class_scale)([input_image, pred_yolo_3, true_yolo_3, true_boxes])
	train_model = Model([input_image, true_boxes, true_yolo_1, true_yolo_2, true_yolo_3], [loss_yolo_1, loss_yolo_2, loss_yolo_3])
	infer_model = Model(input_image, [pred_yolo_1, pred_yolo_2, pred_yolo_3])
	return [train_model, infer_model]

def dummy_loss(y_true, y_pred):
	return tf.sqrt(tf.reduce_sum(y_pred))

def multi_gpu_model(model, gpus):
	if isinstance(gpus, (list, tuple)):
		num_gpus = len(gpus)
		target_gpu_ids = gpus
	else:
		num_gpus = gpus
		target_gpu_ids = range(num_gpus)
	def get_slice(data, i, parts):
		shape = tf.shape(data)
		batch_size = shape[:1]
		input_shape = shape[1:]
		step = batch_size // parts
		if i == num_gpus - 1:
			size = batch_size - step * i
		else:
			size = step
		size = tf.concat([size, input_shape], axis=0)
		stride = tf.concat([step, input_shape * 0], axis=0)
		start = stride * i
		return tf.slice(data, start, size)
	all_outputs = []
	for i in range(len(model.outputs)):
		all_outputs.append([])
	# Place a copy of the model on each GPU,
	# each getting a slice of the inputs.
	for i, gpu_id in enumerate(target_gpu_ids):
		with tf.device('/gpu:%d' % gpu_id):
			with tf.name_scope('replica_%d' % gpu_id):
				inputs = []
				# Retrieve a slice of the input.
				for x in model.inputs:
					input_shape = tuple(x.get_shape().as_list())[1:]
					slice_i = Lambda(get_slice, output_shape=input_shape, arguments={'i': i, 'parts': num_gpus})(x)
					inputs.append(slice_i)
				# Apply model on slice
				# (creating a model replica on the target device).
				outputs = model(inputs)
				if not isinstance(outputs, list):
					outputs = [outputs]
				# Save the outputs for merging back together later.
				for o in range(len(outputs)):
					all_outputs[o].append(outputs[o])
	# Merge outputs on CPU.
	with tf.device('/cpu:0'):
		merged = []
		for name, outputs in zip(model.output_names, all_outputs):
			merged.append(concatenate(outputs, axis=0, name=name))
		return Model(model.inputs, merged)

def get_color(label):
	'''
	Return a color from a set of predefined colors. Contains 80 colors in total.
	code originally from https://github.com/fizyr/keras-retinanet/
	Args
		label: The label to get the color for.
	Returns
		A list of three values representing a RGB color.
	'''
	if label < len(colors):
		return colors[label]
	else:
		print('Label {} has no color, returning default.'.format(label))
		return (0, 255, 0)

colors = 	[[31 , 0   , 255] ,
			[0   , 159 , 255] ,
			[255 , 95  , 0]   ,
			[255 , 19  , 0]   ,
			[255 , 0   , 0]   ,
			[255 , 38  , 0]   ,
			[0   , 255 , 25]  ,
			[255 , 0   , 133] ,
			[255 , 172 , 0]   ,
			[108 , 0   , 255] ,
			[0   , 82  , 255] ,
			[0   , 255 , 6]   ,
			[255 , 0   , 152] ,
			[223 , 0   , 255] ,
			[12  , 0   , 255] ,
			[0   , 255 , 178] ,
			[108 , 255 , 0]   ,
			[184 , 0   , 255] ,
			[255 , 0   , 76]  ,
			[146 , 255 , 0]   ,
			[51  , 0   , 255] ,
			[0   , 197 , 255] ,
			[255 , 248 , 0]   ,
			[255 , 0   , 19]  ,
			[255 , 0   , 38]  ,
			[89  , 255 , 0]   ,
			[127 , 255 , 0]   ,
			[255 , 153 , 0]   ,
			[0   , 255 , 255] ,
			[0   , 255 , 216] ,
			[0   , 255 , 121] ,
			[255 , 0   , 248] ,
			[70  , 0   , 255] ,
			[0   , 255 , 159] ,
			[0   , 216 , 255] ,
			[0   , 6   , 255] ,
			[0   , 63  , 255] ,
			[31  , 255 , 0]   ,
			[255 , 57  , 0]   ,
			[255 , 0   , 210] ,
			[0   , 255 , 102] ,
			[242 , 255 , 0]   ,
			[255 , 191 , 0]   ,
			[0   , 255 , 63]  ,
			[255 , 0   , 95]  ,
			[146 , 0   , 255] ,
			[184 , 255 , 0]   ,
			[255 , 114 , 0]   ,
			[0   , 255 , 235] ,
			[255 , 229 , 0]   ,
			[0   , 178 , 255] ,
			[255 , 0   , 114] ,
			[255 , 0   , 57]  ,
			[0   , 140 , 255] ,
			[0   , 121 , 255] ,
			[12  , 255 , 0]   ,
			[255 , 210 , 0]   ,
			[0   , 255 , 44]  ,
			[165 , 255 , 0]   ,
			[0   , 25  , 255] ,
			[0   , 255 , 140] ,
			[0   , 101 , 255] ,
			[0   , 255 , 82]  ,
			[223 , 255 , 0]   ,
			[242 , 0   , 255] ,
			[89  , 0   , 255] ,
			[165 , 0   , 255] ,
			[70  , 255 , 0]   ,
			[255 , 0   , 172] ,
			[255 , 76  , 0]   ,
			[203 , 255 , 0]   ,
			[204 , 0   , 255] ,
			[255 , 0   , 229] ,
			[255 , 133 , 0]   ,
			[127 , 0   , 255] ,
			[0   , 235 , 255] ,
			[0   , 255 , 197] ,
			[255 , 0   , 191] ,
			[0   , 44  , 255] ,
			[50  , 255 , 0]]

class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.c       = c
		self.classes = classes
		self.label = -1
		self.score = -1
	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)
		return self.label
	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]
		return self.score

def _interval_overlap(interval_a, interval_b):
	x1, x2 = interval_a
	x3, x4 = interval_b
	if x3 < x1:
		if x4 < x1:
			return 0
		else:
			return min(x2,x4) - x1
	else:
		if x2 < x3:
				return 0
		else:
			return min(x2,x4) - x3

def bbox_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
	intersect = intersect_w * intersect_h
	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	union = w1*h1 + w2*h2 - intersect
	return float(intersect) / union

def draw_boxes(image, boxes, labels, obj_thresh, quiet=True):
	for box in boxes:
		label_str = ''
		label = -1
		for i in range(len(labels)):
			if box.classes[i] > obj_thresh:
				if label_str != '': label_str += ', '
				label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')
				label = i
			if not quiet: print(label_str)
		if label >= 0:
			text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
			width, height = text_size[0][0], text_size[0][1]
			region = np.array([[box.xmin-3, box.ymin], [box.xmin-3, box.ymin-height-26], [box.xmin+width+13, box.ymin-height-26], [box.xmin+width+13, box.ymin]], dtype='int32')
			cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=get_color(label), thickness=5)
			cv2.fillPoly(img=image, pts=[region], color=get_color(label))
			cv2.putText(img=image, text=label_str, org=(box.xmin+13, box.ymin - 13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(0,0,0), thickness=2)
	return image

def _sigmoid(x):
	return scipy.special.expit(x)

def makedirs(path):
	try:
		os.makedirs(path)
	except OSError:
		if not os.path.isdir(path):
			raise

def evaluate(model, generator, iou_threshold=0.5, obj_thresh=0.5, nms_thresh=0.45, net_h=416, net_w=416, save_path=None):
	'''
	Evaluate a given dataset using a given model.
	code originally from https://github.com/fizyr/keras-retinanet
	# Arguments
		model           : The model to evaluate.
		generator       : The generator that represents the dataset to evaluate.
		iou_threshold   : The threshold used to consider when a detection is positive or negative.
		obj_thresh      : The threshold used to distinguish between object and non-object
		nms_thresh      : The threshold used to determine whether two detections are duplicates
		net_h           : The height of the input image to the model, higher value results in better accuracy
		net_w           : The width of the input image to the model
		save_path       : The path to save images with visualized detections to.
	# Returns
		A dict mapping class names to mAP scores.
	'''
	# gather all detections and annotations
	all_detections      = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
	all_annotations     = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
	for i in range(generator.size()):
		raw_image = [generator.load_image(i)]
		# make the boxes and the labels
		pred_boxes = get_yolo_boxes(model, raw_image, net_h, net_w, generator.get_anchors(), obj_thresh, nms_thresh)[0]
		score = np.array([box.get_score() for box in pred_boxes])
		pred_labels = np.array([box.label for box in pred_boxes])
		if len(pred_boxes) > 0:
			pred_boxes = np.array([[box.xmin, box.ymin, box.xmax, box.ymax, box.get_score()] for box in pred_boxes]) 
		else:
			pred_boxes = np.array([[]])
		# sort the boxes and the labels according to scores
		score_sort  = np.argsort(-score)
		pred_labels = pred_labels[score_sort]
		pred_boxes  = pred_boxes[score_sort]
		# copy detections to all_detections
		for label in range(generator.num_classes()):
			all_detections[i][label] = pred_boxes[pred_labels == label, :]
		annotations = generator.load_annotation(i)
		# copy detections to all_annotations
		for label in range(generator.num_classes()):
			all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
	# compute mAP by comparing all detections and all annotations
	average_precisions = {}
	for label in range(generator.num_classes()):
		false_positives = np.zeros((0,))
		true_positives  = np.zeros((0,))
		scores          = np.zeros((0,))
		num_annotations = 0.0
		for i in range(generator.size()):
			detections              = all_detections[i][label]
			annotations             = all_annotations[i][label]
			num_annotations        += annotations.shape[0]
			detected_annotations    = []
			for d in detections:
				scores = np.append(scores, d[4])
				if annotations.shape[0] == 0:
					false_positives = np.append(false_positives, 1)
					true_positives  = np.append(true_positives, 0)
					continue
				overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
				assigned_annotation = np.argmax(overlaps, axis=1)
				max_overlap         = overlaps[0, assigned_annotation]
				if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
					false_positives = np.append(false_positives, 0)
					true_positives  = np.append(true_positives, 1)
					detected_annotations.append(assigned_annotation)
				else:
					false_positives = np.append(false_positives, 1)
					true_positives  = np.append(true_positives, 0)
		# no annotations -> AP for this class is 0 (is this correct?)
		if num_annotations == 0:
			average_precisions[label] = 0
			continue
		# sort by score
		indices         = np.argsort(-scores)
		false_positives = false_positives[indices]
		true_positives  = true_positives[indices]
		# compute false positives and true positives
		false_positives = np.cumsum(false_positives)
		true_positives  = np.cumsum(true_positives)
		# compute recall and precision
		recall          = true_positives / num_annotations
		precision       = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
		# compute average precision
		average_precision           = compute_ap(recall, precision)
		average_precisions[label]   = average_precision
	return average_precisions

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
	if (float(net_w)/image_w) < (float(net_h)/image_h):
		new_w = net_w
		new_h = (image_h*net_w)/image_w
	else:
		new_h = net_w
		new_w = (image_w*net_h)/image_h
	for i in range(len(boxes)):
		x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
		y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
		boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
		boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
		boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
		boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def do_nms(boxes, nms_thresh):
	if len(boxes) > 0:
		nb_class = len(boxes[0].classes)
	else:
		return
	for c in range(nb_class):
		sorted_indices = np.argsort([-box.classes[c] for box in boxes])
		for i in range(len(sorted_indices)):
			index_i = sorted_indices[i]
			if boxes[index_i].classes[c] == 0: continue
			for j in range(i+1, len(sorted_indices)):
				index_j = sorted_indices[j]
				if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
					boxes[index_j].classes[c] = 0

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
	grid_h, grid_w = netout.shape[:2]
	nb_box = 3
	netout = netout.reshape((grid_h, grid_w, nb_box, -1))
	nb_class = netout.shape[-1] - 5
	boxes = []
	netout[..., :2]     = _sigmoid(netout[..., :2])
	netout[..., 4]      = _sigmoid(netout[..., 4])
	netout[..., 5:]     = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
	netout[..., 5:]    *= netout[..., 5:] > obj_thresh
	for i in range(grid_h*grid_w):
		row = i // grid_w
		col = i % grid_w
		for b in range(nb_box):
			# 4th element is objectness score
			objectness = netout[row, col, b, 4]
			if(objectness <= obj_thresh): continue
			# first 4 elements are x, y, w, and h
			x, y, w, h = netout[row,col,b,:4]
			x = (col + x) / grid_w # center position, unit: image width
			y = (row + y) / grid_h # center position, unit: image height
			w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
			h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
			# last elements are class probabilities
			classes = netout[row,col,b,5:]
			box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
			boxes.append(box)
	return boxes

def preprocess_input(image, net_h, net_w):
	new_h, new_w, _ = image.shape
	# determine the new size of the image
	if (float(net_w)/new_w) < (float(net_h)/new_h):
		new_h = (new_h * net_w)//new_w
		new_w = net_w
	else:
		new_w = (new_w * net_h)//new_h
		new_h = net_h
	# resize the image to the new size
	resized = cv2.resize(image[:,:,::-1]/255., (new_w, new_h))
	# embed the image into the standard letter box
	new_image = np.ones((net_h, net_w, 3)) * 0.5
	new_image[(net_h-new_h)//2:(net_h+new_h)//2, (net_w-new_w)//2:(net_w+new_w)//2, :] = resized
	new_image = np.expand_dims(new_image, 0)
	return new_image

def normalize(image):
	return image/255.

def get_yolo_boxes(model, images, net_h, net_w, anchors, obj_thresh, nms_thresh):
	image_h, image_w, _ = images[0].shape
	nb_images           = len(images)
	batch_input         = np.zeros((nb_images, net_h, net_w, 3))
	# preprocess the input
	for i in range(nb_images):
		batch_input[i] = preprocess_input(images[i], net_h, net_w)
	# run the prediction
	batch_output = model.predict_on_batch(batch_input)
	batch_boxes  = [None]*nb_images
	for i in range(nb_images):
		yolos = [batch_output[0][i], batch_output[1][i], batch_output[2][i]]
		boxes = []
		# decode the output of the network
		for j in range(len(yolos)):
			yolo_anchors = anchors[(2-j)*6:(3-j)*6] # config['model']['anchors']
			boxes += decode_netout(yolos[j], yolo_anchors, obj_thresh, net_h, net_w)
		# correct the sizes of the bounding boxes
		correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
		# suppress non-maximal boxes
		do_nms(boxes, nms_thresh)
		batch_boxes[i] = boxes
	return batch_boxes

def compute_overlap(a, b):
	'''
	Code originally from https://github.com/rbgirshick/py-faster-rcnn.
	Parameters
	----------
	a: (N, 4) ndarray of float
	b: (K, 4) ndarray of float
	Returns
	-------
	overlaps: (N, K) ndarray of overlap between boxes and query_boxes
	'''
	area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
	iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
	ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])
	iw = np.maximum(iw, 0)
	ih = np.maximum(ih, 0)
	ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
	ua = np.maximum(ua, np.finfo(float).eps)
	intersection = iw * ih
	return intersection / ua

def compute_ap(recall, precision):
	'''
	Compute the average precision, given the recall and precision curves.
	Code originally from https://github.com/rbgirshick/py-faster-rcnn.
	# Arguments
		recall:    The recall curve (list).
		precision: The precision curve (list).
	# Returns
		The average precision as computed in py-faster-rcnn.
	'''
	# correct AP calculation
	# first append sentinel values at the end
	mrec = np.concatenate(([0.], recall, [1.]))
	mpre = np.concatenate(([0.], precision, [0.]))
	# compute the precision envelope
	for i in range(mpre.size - 1, 0, -1):
		mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
	# to calculate area under PR curve, look for points
	# where X axis (recall) changes value
	i = np.where(mrec[1:] != mrec[:-1])[0]
	# and sum (\Delta recall) * prec
	ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
	return ap

def _softmax(x, axis=-1):
	x = x - np.amax(x, axis, keepdims=True)
	e_x = np.exp(x)
	return e_x / e_x.sum(axis, keepdims=True)

class BatchGenerator(Sequence):
	def __init__(self,
		instances,
		anchors,
		labels,
		downsample=32, # ratio between network input's size and network output's size, 32 for YOLOv3
		max_box_per_image=30,
		batch_size=1,
		min_net_size=320,
		max_net_size=608,
		shuffle=True,
		jitter=True,
		norm=None):
		self.instances          = instances
		self.batch_size         = batch_size
		self.labels             = labels
		self.downsample         = downsample
		self.max_box_per_image  = max_box_per_image
		self.min_net_size       = (min_net_size // self.downsample) * self.downsample
		self.max_net_size       = (max_net_size // self.downsample) * self.downsample
		self.shuffle            = shuffle
		self.jitter             = jitter
		self.norm               = norm
		self.anchors            = [BoundBox(0, 0, anchors[2*i], anchors[2*i+1]) for i in range(len(anchors)//2)]
		self.net_h              = 416
		self.net_w              = 416
		if shuffle: np.random.shuffle(self.instances)
	def __len__(self):
		return int(np.ceil(float(len(self.instances))/self.batch_size))
	def __getitem__(self, idx):
		# get image input size, change every 10 batches
		net_h, net_w = self._get_net_size(idx)
		base_grid_h, base_grid_w = net_h//self.downsample, net_w//self.downsample
		# determine the first and the last indices of the batch
		l_bound = idx*self.batch_size
		r_bound = (idx+1)*self.batch_size
		if r_bound > len(self.instances):
			r_bound = len(self.instances)
			l_bound = r_bound - self.batch_size
		x_batch = np.zeros((r_bound - l_bound, net_h, net_w, 3)) # input images
		t_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.max_box_per_image, 4)) # list of groundtruth boxes
		# initialize the inputs and the outputs
		yolo_1 = np.zeros((r_bound - l_bound, 1*base_grid_h, 1*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels))) # desired network output 1
		yolo_2 = np.zeros((r_bound - l_bound, 2*base_grid_h, 2*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels))) # desired network output 2
		yolo_3 = np.zeros((r_bound - l_bound, 4*base_grid_h, 4*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels))) # desired network output 3
		yolos = [yolo_3, yolo_2, yolo_1]
		dummy_yolo_1 = np.zeros((r_bound - l_bound, 1))
		dummy_yolo_2 = np.zeros((r_bound - l_bound, 1))
		dummy_yolo_3 = np.zeros((r_bound - l_bound, 1))
		instance_count = 0
		true_box_index = 0
		# do the logic to fill in the inputs and the output
		for train_instance in self.instances[l_bound:r_bound]:
			# augment input image and fix object's position and size
			img, all_objs = self._aug_image(train_instance, net_h, net_w)
			for obj in all_objs:
				# find the best anchor box for this object
				max_anchor  = None
				max_index   = -1
				max_iou     = -1
				shifted_box = BoundBox(0, 0, obj['xmax']-obj['xmin'], obj['ymax']-obj['ymin'])
				for i in range(len(self.anchors)):
					anchor  = self.anchors[i]
					iou     = bbox_iou(shifted_box, anchor)
					if max_iou < iou:
						max_anchor  = anchor
						max_index   = i
						max_iou     = iou
				# determine the yolo to be responsible for this bounding box
				yolo = yolos[max_index//3]
				grid_h, grid_w = yolo.shape[1:3]
				# determine the position of the bounding box on the grid
				center_x = .5*(obj['xmin'] + obj['xmax'])
				center_x = center_x / float(net_w) * grid_w # sigma(t_x) + c_x
				center_y = .5*(obj['ymin'] + obj['ymax'])
				center_y = center_y / float(net_h) * grid_h # sigma(t_y) + c_y
				# determine the sizes of the bounding box
				w = np.log((obj['xmax'] - obj['xmin']) / float(max_anchor.xmax)) # t_w
				h = np.log((obj['ymax'] - obj['ymin']) / float(max_anchor.ymax)) # t_h
				box = [center_x, center_y, w, h]
				# determine the index of the label
				obj_indx = self.labels.index(obj['name'])
				# determine the location of the cell responsible for this object
				grid_x = int(np.floor(center_x))
				grid_y = int(np.floor(center_y))
				# assign ground truth x, y, w, h, confidence and class probs to y_batch
				yolo[instance_count, grid_y, grid_x, max_index%3]      = 0
				yolo[instance_count, grid_y, grid_x, max_index%3, 0:4] = box
				yolo[instance_count, grid_y, grid_x, max_index%3, 4  ] = 1.
				yolo[instance_count, grid_y, grid_x, max_index%3, 5+obj_indx] = 1
				# assign the true box to t_batch
				true_box = [center_x, center_y, obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin']]
				t_batch[instance_count, 0, 0, 0, true_box_index] = true_box
				true_box_index += 1
				true_box_index = true_box_index % self.max_box_per_image
			# assign input image to x_batch
			if self.norm != None:
				x_batch[instance_count] = self.norm(img)
			else:
				# plot image and bounding boxes for sanity check
				for obj in all_objs:
					cv2.rectangle(img, (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
					cv2.putText(img, obj['name'], (obj['xmin']+2, obj['ymin']+12), 0, 1.2e-3 * img.shape[0], (0,255,0), 2)
				x_batch[instance_count] = img
			# increase instance counter in the current batch
			instance_count += 1
		return [x_batch, t_batch, yolo_1, yolo_2, yolo_3], [dummy_yolo_1, dummy_yolo_2, dummy_yolo_3]
	def _get_net_size(self, idx):
		if idx%10 == 0:
			net_size = self.downsample*np.random.randint(self.min_net_size/self.downsample, self.max_net_size/self.downsample+1)
			print('resizing: ', net_size, net_size)
			self.net_h, self.net_w = net_size, net_size
		return self.net_h, self.net_w
	def _aug_image(self, instance, net_h, net_w):
		image_name = instance['filename']
		image = cv2.imread(image_name) # RGB image
		if image is None: print('Cannot find ', image_name)
		image = image[:,:,::-1] # RGB image
		image_h, image_w, _ = image.shape
		# determine the amount of scaling and cropping
		dw = self.jitter * image_w;
		dh = self.jitter * image_h;
		new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh));
		scale = np.random.uniform(0.25, 2);
		if (new_ar < 1):
			new_h = int(scale * net_h);
			new_w = int(net_h * new_ar);
		else:
			new_w = int(scale * net_w);
			new_h = int(net_w / new_ar);
		dx = int(np.random.uniform(0, net_w - new_w));
		dy = int(np.random.uniform(0, net_h - new_h));
		# apply scaling and cropping
		im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)
		# randomly distort hsv space
		im_sized = random_distort_image(im_sized)
		# randomly flip
		flip = np.random.randint(2)
		im_sized = random_flip(im_sized, flip)
		# correct the size and pos of bounding boxes
		all_objs = correct_bounding_boxes(instance['object'], new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h)
		return im_sized, all_objs
	def on_epoch_end(self):
		if self.shuffle: np.random.shuffle(self.instances)
	def num_classes(self):
		return len(self.labels)
	def size(self):
		return len(self.instances)
	def get_anchors(self):
		anchors = []
		for anchor in self.anchors:
			anchors += [anchor.xmax, anchor.ymax]
		return anchors
	def load_annotation(self, i):
		annots = []
		for obj in self.instances[i]['object']:
			annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.labels.index(obj['name'])]
			annots += [annot]
		if len(annots) == 0: annots = [[]]
		return np.array(annots)
	def load_image(self, i):
		return cv2.imread(self.instances[i]['filename'])
def parse_voc_annotation(ann_dir, img_dir, cache_name, labels=[]):
	if os.path.exists(cache_name):
		with open(cache_name, 'rb') as handle:
			cache = pickle.load(handle)
		all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
	else:
		all_insts = []
		seen_labels = {}
		for ann in sorted(os.listdir(ann_dir)):
			img = {'object':[]}
			try:
				tree = ET.parse(ann_dir + ann)
			except Exception as e:
				print(e)
				print('Ignore this bad annotation: ' + ann_dir + ann)
				continue
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
				all_insts += [img]
		cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
		with open(cache_name, 'wb') as handle:
			pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
	return all_insts, seen_labels

class CustomTensorBoard(TensorBoard):
	'''
	To log the loss after each batch
	'''
	def __init__(self, log_every=1, **kwargs):
		super(CustomTensorBoard, self).__init__(**kwargs)
		self.log_every = log_every
		self.counter = 0
	def on_batch_end(self, batch, logs=None):
		self.counter+=1
		if self.counter%self.log_every==0:
			for name, value in logs.items():
				if name in ['batch', 'size']:
					continue
				summary = tf.Summary()
				summary_value = summary.value.add()
				summary_value.simple_value = value.item()
				summary_value.tag = name
				self.writer.add_summary(summary, self.counter)
			self.writer.flush()
		super(CustomTensorBoard, self).on_batch_end(batch, logs)

class CustomModelCheckpoint(ModelCheckpoint):
	'''
	To save the template model, not the multi-GPU model
	'''
	def __init__(self, model_to_save, **kwargs):
		super(CustomModelCheckpoint, self).__init__(**kwargs)
		self.model_to_save = model_to_save
	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}
		self.epochs_since_last_save += 1
		if self.epochs_since_last_save >= self.period:
			self.epochs_since_last_save = 0
			filepath = self.filepath.format(epoch=epoch + 1, **logs)
			if self.save_best_only:
				current = logs.get(self.monitor)
				if current is None:
					warnings.warn('Can save best model only with %s available, '
									'skipping.' % (self.monitor), RuntimeWarning)
				else:
					if self.monitor_op(current, self.best):
						if self.verbose > 0:
							print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
									' saving model to %s' % (epoch + 1, self.monitor, self.best, current, filepath))
						self.best = current
						if self.save_weights_only:
							self.model_to_save.save_weights(filepath, overwrite=True)
						else:
							self.model_to_save.save(filepath, overwrite=True)
					else:
						if self.verbose > 0:
							print('\nEpoch %05d: %s did not improve from %0.5f' % (epoch + 1, self.monitor, self.best))
			else:
				if self.verbose > 0:
					print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
				if self.save_weights_only:
					self.model_to_save.save_weights(filepath, overwrite=True)
				else:
					self.model_to_save.save(filepath, overwrite=True)
		super(CustomModelCheckpoint, self).on_batch_end(epoch, logs)

def create_training_instances(train_annot_folder, train_image_folder, train_cache, valid_annot_folder, valid_image_folder, valid_cache, labels,):
	# parse annotations of the training set
	train_ints, train_labels = parse_voc_annotation(train_annot_folder, train_image_folder, train_cache, labels)
	# parse annotations of the validation set, if any, otherwise split the training set
	if os.path.exists(valid_annot_folder):
		valid_ints, valid_labels = parse_voc_annotation(valid_annot_folder, valid_image_folder, valid_cache, labels)
	else:
		print('valid_annot_folder not exists. Spliting the trainining set.')
		train_valid_split = int(0.8*len(train_ints))
		np.random.seed(0)
		np.random.shuffle(train_ints)
		np.random.seed()
		valid_ints = train_ints[train_valid_split:]
		train_ints = train_ints[:train_valid_split]
	# compare the seen labels with the given labels in config.json
	if len(labels) > 0:
		overlap_labels = set(labels).intersection(set(train_labels.keys()))
		print('Seen labels: \t'  + str(train_labels) + '\n')
		print('Given labels: \t' + str(labels))
		# return None, None, None if some given label is not in the dataset
		if len(overlap_labels) < len(labels):
			print('Some labels have no annotations! Please revise the list of labels in the config.json.')
			return None, None, None
	else:
		print('No labels are provided. Train on all seen labels.')
		print(train_labels)
		labels = train_labels.keys()
	max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])
	return train_ints, valid_ints, sorted(labels), max_box_per_image

def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save):
	makedirs(tensorboard_logs)
	early_stop = EarlyStopping(
		monitor         = 'loss',
		min_delta       = 0.01,
		patience        = 5,
		mode            = 'min',
		verbose         = 1)
	checkpoint = CustomModelCheckpoint(
		model_to_save   = model_to_save,
		filepath        = saved_weights_name,# + '{epoch:02d}.h5',
		monitor         = 'loss',
		verbose         = 1,
		save_best_only  = True,
		mode            = 'min',
		period          = 1)
	reduce_on_plateau = ReduceLROnPlateau(
		monitor         = 'loss',
		factor          = .1,
		patience        = 2,
		verbose         = 1,
		mode            = 'min',
		epsilon         = 0.01,
		cooldown        = 0,
		min_lr          = 0)
	tensorboard = CustomTensorBoard(
		log_dir         = tensorboard_logs,
		write_graph     = True,
		write_images    = True,)
	return [early_stop, checkpoint, reduce_on_plateau, tensorboard]

def create_model(
	nb_class,
	anchors,
	max_box_per_image,
	max_grid, batch_size,
	warmup_batches,
	ignore_thresh,
	multi_gpu,
	saved_weights_name,
	lr,
	grid_scales,
	obj_scale,
	noobj_scale,
	xywh_scale,
	class_scale):
	if multi_gpu > 1:
		with tf.device('/cpu:0'):
			template_model, infer_model = create_yolov3_model(
				nb_class            = nb_class,
				anchors             = anchors,
				max_box_per_image   = max_box_per_image,
				max_grid            = max_grid,
				batch_size          = batch_size//multi_gpu,
				warmup_batches      = warmup_batches,
				ignore_thresh       = ignore_thresh,
				grid_scales         = grid_scales,
				obj_scale           = obj_scale,
				noobj_scale         = noobj_scale,
				xywh_scale          = xywh_scale,
				class_scale         = class_scale)
	else:
		template_model, infer_model = create_yolov3_model(
			nb_class                = nb_class,
			anchors                 = anchors,
			max_box_per_image       = max_box_per_image,
			max_grid                = max_grid,
			batch_size              = batch_size,
			warmup_batches          = warmup_batches,
			ignore_thresh           = ignore_thresh,
			grid_scales             = grid_scales,
			obj_scale               = obj_scale,
			noobj_scale             = noobj_scale,
			xywh_scale              = xywh_scale,
			class_scale             = class_scale)
	# load the pretrained weight if exists, otherwise load the backend weight only
	if os.path.exists(saved_weights_name):
		print('\nLoading pretrained weights.\n')
		template_model.load_weights(saved_weights_name)
	#else:
		#template_model.load_weights('backend.h5', by_name=True)
	if multi_gpu > 1:
		train_model = multi_gpu_model(template_model, gpus=multi_gpu)
	else:
		train_model = template_model
	optimizer = Adam(lr=lr, clipnorm=0.001)
	train_model.compile(loss=dummy_loss, optimizer=optimizer)
	return train_model, infer_model

def main_train():
#	config_path = config#'./config.json'
#	with open(config_path) as config_buffer:
#		config = json.loads(config_buffer.read())
	'''
	Parse the annotations
	'''
	train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
		config['train']['train_annot_folder'],
		config['train']['train_image_folder'],
		config['train']['cache_name'],
		config['valid']['valid_annot_folder'],
		config['valid']['valid_image_folder'],
		config['valid']['cache_name'],
		config['model']['labels'])
	print('\nTraining on: \t' + str(labels) + '\n')
	'''
	Create the generators
	'''
	train_generator = BatchGenerator(
		instances           = train_ints,
		anchors             = config['model']['anchors'],
		labels              = labels,
		downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
		max_box_per_image   = max_box_per_image,
		batch_size          = config['train']['batch_size'],
		min_net_size        = config['model']['min_input_size'],
		max_net_size        = config['model']['max_input_size'],
		shuffle             = True,
		jitter              = 0.3,
		norm                = normalize)
	valid_generator = BatchGenerator(
		instances           = valid_ints,
		anchors             = config['model']['anchors'],
		labels              = labels,
		downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
		max_box_per_image   = max_box_per_image,
		batch_size          = config['train']['batch_size'],
		min_net_size        = config['model']['min_input_size'],
		max_net_size        = config['model']['max_input_size'],
		shuffle             = True,
		jitter              = 0.0,
		norm                = normalize)
	'''
	Create the model
	'''
	if os.path.exists(config['train']['saved_weights_name']):
		config['train']['warmup_epochs'] = 0
	warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times']*len(train_generator))
	os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
	multi_gpu = len(config['train']['gpus'].split(','))
	train_model, infer_model = create_model(
		nb_class             = len(labels),
		anchors              = config['model']['anchors'],
		max_box_per_image    = max_box_per_image,
		max_grid             = [config['model']['max_input_size'], config['model']['max_input_size']],
		batch_size           = config['train']['batch_size'],
		warmup_batches       = warmup_batches,
		ignore_thresh        = config['train']['ignore_thresh'],
		multi_gpu            = multi_gpu,
		saved_weights_name   = config['train']['saved_weights_name'],
		lr                   = config['train']['learning_rate'],
		grid_scales          = config['train']['grid_scales'],
		obj_scale            = config['train']['obj_scale'],
		noobj_scale          = config['train']['noobj_scale'],
		xywh_scale           = config['train']['xywh_scale'],
		class_scale          = config['train']['class_scale'],)
	'''
	Kick off the training
	'''
	callbacks = create_callbacks(config['train']['saved_weights_name'], config['train']['tensorboard_dir'], infer_model)
	train_model.fit_generator(
		generator           = train_generator,
		steps_per_epoch     = len(train_generator) * config['train']['train_times'],
		epochs              = config['train']['nb_epochs'] + config['train']['warmup_epochs'],
		verbose             = 2 if config['train']['debug'] else 1,
		callbacks           = callbacks,
		workers             = 4,
		max_queue_size      = 8)
	# make a GPU version of infer_model for evaluation
	if multi_gpu > 1:
		infer_model = load_model(config['train']['saved_weights_name'])
	'''
	Run the evaluation
	'''
	# compute mAP for all the classes
	average_precisions = evaluate(infer_model, valid_generator)
	# print the score
	for label, average_precision in average_precisions.items():
		print(labels[label] + ': {:.4f}'.format(average_precision))
	print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))

def main_predict(FILENAME):
	config_path		= config#'./config.json'
	input_path		= FILENAME
#	with open(config_path) as config_buffer:
#		config = json.load(config_buffer)
	# Set some parameter
	net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
	obj_thresh, nms_thresh = 0.5, 0.45
	# Load the model
	os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
	infer_model = load_model(config['train']['saved_weights_name'])
	# Predict bounding boxes
	image_paths = []
	if os.path.isdir(input_path):
		for inp_file in os.listdir(input_path):
			image_paths += [input_path + inp_file]
	else:
		image_paths += [input_path]
	image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]
	# the main loop
	for image_path in image_paths:
		image = cv2.imread(image_path)
		print(image_path)
		# predict the bounding boxes
		boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]
		# draw bounding boxes on the image using labels
		draw_boxes(image, boxes, config['model']['labels'], obj_thresh)
		# write the image with bounding boxes to file
		cv2.imwrite('out_' + image_path.split('/')[-1], np.uint8(image))

if __name__ == '__main__':
	if sys.argv[1] == '-t':
	    main_train()
	elif sys.argv[1] == '-d':
		main_predict(sys.argv[2])
	else:
		print('Error in command')
