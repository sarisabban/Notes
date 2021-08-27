#!/usr/bin/python3

'''
Reading:
https://www.dlology.com/blog/gentle-guide-on-how-yolo-object-localization-works-with-keras/
https://www.dlology.com/blog/gentle-guide-on-how-yolo-object-localization-works-with-keras-part-2/
https://www.analyticsvidhya.com/blog/2018/06/understanding-building-object-detection-model-python/
https://github.com/fizyr/keras-retinanet
https://medium.com/smileinnovation/capturing-your-dinner-a-deep-learning-story-bf8f8b65f26f
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#6
https://github.com/experiencor/keras-yolo2/blob/master/examples/Blood%20Cell%20Detection.ipynb
https://pjreddie.com/media/files/yolov2.weights
'''

import numpy as np
import keras
import os

# Initial Setup
LABELS				=['RBC']							# Image label
IMAGE_H, IMAGE_W	= 416, 416							# Image size in px
GRID_H, GRID_W		= 13, 13							# Segment Image into this number of segments per row and per column (size of feature map sub-images)
BOX					= 5									# Number of bounding boxes a cell must intersect on the feature map.
CLASS				= len(LABELS)						# Number of classes
CLASS_WEIGHTS		= np.ones(CLASS, dtype='float32')	# Class weight (1 for True)
ANCHORS				= [	0.57273, 0.677385, 1.87446,
						2.06253, 3.33843, 5.47434,
						7.88282, 3.52778, 9.77052,
						9.16828]						# Box initial sizes (anchor boxes) to detect multiple object in the same region, the network will just adjust the nearest anchor box to the object
OBJ_THRESHOLD		= 0.3								# Confidence score that a bounding box incloses a desired object, if less discard bounding box
NMS_THRESHOLD		= 0.3								# Non-max suppression (threshold to detec an object only once, rather than the same object multiple times)
NO_OBJECT_SCALE		= 1.0								# 
OBJECT_SCALE		= 5.0								# 
COORD_SCALE			= 1.0								# 
CLASS_SCALE			= 1.0								# 
BATCH_SIZE			= 16								# Training batches
WARM_UP_BATCHES		= 100								# 
TRUE_BOX_BUFFER		= 50								# 
image_path			= './dataset/JPEGImages/'			# Dataset image location
annot_path			= './dataset/Annotations/'			# Dataset annotation location

# Configuration of the generator
generator_config = {	'IMAGE_H'			: IMAGE_H,
						'IMAGE_W'			: IMAGE_W,
						'GRID_H'			: GRID_H,
						'GRID_W'			: GRID_W,
						'BOX'				: BOX,
						'LABELS'			: LABELS,
						'CLASS'				: len(LABELS),
						'ANCHORS'			: ANCHORS,
						'BATCH_SIZE'		: BATCH_SIZE,
						'TRUE_BOX_BUFFER'	: 50,}

import xml.etree.ElementTree as ET
from keras.utils import Sequence
from utils import BoundBox, bbox_iou
from imgaug import augmenters as iaa
import cv2
import copy
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda

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

class BatchGenerator(Sequence):
	def __init__(self, images,
						config,
						shuffle=True,
						jitter=True,
						norm=None):
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

for img in all_imgs:
	img['filename'] = img['filename']+'.jpg'

batches = BatchGenerator(all_imgs, generator_config)
image = batches[0][0][0][0]

def normalize(image):
	return image/255.

train_valid_split = int(0.8*len(all_imgs))
train_batch = BatchGenerator(all_imgs[:train_valid_split], generator_config)
valid_batch = BatchGenerator(all_imgs[train_valid_split:], generator_config, norm=normalize)

def space_to_depth_x2(x):
	return tf.space_to_depth(x, block_size=2)

input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))





























'''
# Layer 1
x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 2
x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
x = BatchNormalization(name='norm_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 3
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
x = BatchNormalization(name='norm_3')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 4
x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
x = BatchNormalization(name='norm_4')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 5
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
x = BatchNormalization(name='norm_5')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 6
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
x = BatchNormalization(name='norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 7
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
x = BatchNormalization(name='norm_7')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 8
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False, input_shape=(416,416,3))(x)
x = BatchNormalization(name='norm_8')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 9
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
x = BatchNormalization(name='norm_9')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 10
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
x = BatchNormalization(name='norm_10')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 11
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
x = BatchNormalization(name='norm_11')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 12
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
x = BatchNormalization(name='norm_12')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 13
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
x = BatchNormalization(name='norm_13')(x)
x = LeakyReLU(alpha=0.1)(x)

skip_connection = x

x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 14
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
x = BatchNormalization(name='norm_14')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 15
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
x = BatchNormalization(name='norm_15')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 16
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
x = BatchNormalization(name='norm_16')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 17
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
x = BatchNormalization(name='norm_17')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 18
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
x = BatchNormalization(name='norm_18')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 19
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
x = BatchNormalization(name='norm_19')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 20
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
x = BatchNormalization(name='norm_20')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 21
skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
skip_connection = BatchNormalization(name='norm_21')(skip_connection)
skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
skip_connection = Lambda(space_to_depth_x2)(skip_connection)

x = concatenate([skip_connection, x])

# Layer 22
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
x = BatchNormalization(name='norm_22')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 23
x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

# small hack to allow true_boxes to be registered when Keras build the model 
# for more information: https://github.com/fchollet/keras/issues/2790
output = Lambda(lambda args: args[0])([output, true_boxes])

model = Model([input_image, true_boxes], output)





model.summary()























weight_reader = WeightReader(wt_path)



weight_reader.reset()
nb_conv = 23

for i in range(1, nb_conv+1):
    conv_layer = model.get_layer('conv_' + str(i))
    
    if i < nb_conv:
        norm_layer = model.get_layer('norm_' + str(i))
        
        size = np.prod(norm_layer.get_weights()[0].shape)

        beta  = weight_reader.read_bytes(size)
        gamma = weight_reader.read_bytes(size)
        mean  = weight_reader.read_bytes(size)
        var   = weight_reader.read_bytes(size)

        weights = norm_layer.set_weights([gamma, beta, mean, var])       
        
    if len(conv_layer.get_weights()) > 1:
        bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel, bias])
    else:
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel])





layer = model.layers[-4] # the last convolutional layer
weights = layer.get_weights()

new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
new_bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)

layer.set_weights([new_kernel, new_bias])

'''

"""
def custom_loss(y_true, y_pred):
    #mask_shape = tf.shape(y_true)[:4]
    
    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)))
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))

    cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [BATCH_SIZE, 1, 1, 5, 1])
    
    coord_mask = tf.zeros(mask_shape)
    conf_mask  = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)
    
    seen = tf.Variable(0.)
    total_recall = tf.Variable(0.)
    
    #Adjust prediction
    ### adjust x and y      
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
    
    ### adjust w and h
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2])
    
    ### adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])
    
    ### adjust class probabilities
    pred_box_class = y_pred[..., 5:]
    
    #Adjust ground truth
    ### adjust x and y
    true_box_xy = y_true[..., 0:2] # relative position to the containing cell
    
    ### adjust w and h
    true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically
    
    ### adjust confidence
    true_wh_half = true_box_wh / 2.
    true_mins    = true_box_xy - true_wh_half
    true_maxes   = true_box_xy + true_wh_half
    
    pred_wh_half = pred_box_wh / 2.
    pred_mins    = pred_box_xy - pred_wh_half
    pred_maxes   = pred_box_xy + pred_wh_half       
    
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)
    
    true_box_conf = iou_scores * y_true[..., 4]
    
    ### adjust class probabilities
    true_box_class = tf.argmax(y_true[..., 5:], -1)
    
    #Determine the masks
    ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE
    
    ### confidence mask: penelize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]
    
    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half
    
    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)
    
    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half    
    
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE
    
    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE
    
    ### class mask: simply the position of the ground truth boxes (the predictors)
    class_mask = y_true[..., 4] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE       
    
    #Warm-up training
    no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE/2.)
    seen = tf.assign_add(seen, 1.)
    
    true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES), 
                          lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
                                   true_box_wh + tf.ones_like(true_box_wh) * np.reshape(ANCHORS, [1,1,1,BOX,2]) * no_boxes_mask, 
                                   tf.ones_like(coord_mask)],
                          lambda: [true_box_xy, 
                                   true_box_wh,
                                   coord_mask])
    
    #Finalize the loss
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
    
    loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
    
    loss = loss_xy + loss_wh + loss_conf + loss_class
    
    nb_true_box = tf.reduce_sum(y_true[..., 4])
    nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

    #Debugging code
    current_recall = nb_pred_box/(nb_true_box + 1e-6)
    total_recall = tf.assign_add(total_recall, current_recall) 

    loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
    loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)
    
    return loss





early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=3, 
                           mode='min', 
                           verbose=1)

checkpoint = ModelCheckpoint('weights_blood.h5', 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min', 
                             period=1)



#model.load_weights('weights_blood.h5')




tb_counter  = len([log for log in os.listdir(os.path.expanduser('./logs/')) if 'blood' in log]) + 1
tensorboard = TensorBoard(log_dir=os.path.expanduser('./logs/') + 'blood' + '_' + str(tb_counter), 
                          histogram_freq=0, 
                          write_graph=True, 
                          write_images=False)

optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
#optimizer = RMSprop(lr=1e-5, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss=custom_loss, optimizer=optimizer)

model.fit_generator(generator        = train_batch, 
                    steps_per_epoch  = len(train_batch), 
                    epochs           = 100, 
                    verbose          = 1,
                    validation_data  = valid_batch,
                    validation_steps = len(valid_batch),
                    callbacks        = [early_stop, checkpoint, tensorboard], 
                    max_queue_size   = 3)





#model.load_weights("weights_blood.h5")

dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))



image = cv2.imread('./XX/JPEGImages/BloodImage_00032.jpg')
#image = cv2.imread('./COCO_val2014_000000000196.jpg')
#image = cv2.imread(all_imgs[train_valid_split:][28]['filename'])

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
"""
