#-*- coding:utf-8 -*-
import os
import numpy as np
from imp import reload
from PIL import Image, ImageOps

from keras import backend as K
from keras.layers import Input
from keras.models import Model
# import keras.backend as K
from keras.utils import multi_gpu_model

import dl_crnn as densenet

#reload(densenet)

#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
#GPU_NUM = 2
GPU_NUM=1
encode_dct =  {}
char_set = open('/home/denghailong/text_renderer/data/chars/chn.txt', 'r', encoding='utf-8').readlines()
for i in range (0, len(char_set)):
	c = char_set[i].strip('\n')
	encode_dct[c] = i
char_set = ''.join([ch.strip('\n') for ch in char_set] + ['卍'])

#characters = ''.join([chr(i) for i in range(32, 127)] + ['卍'])
nclass = len(char_set)

mult_model, basemodel = densenet.get_model(False, 32, nclass)
#basemodel = Model(inputs=input, outputs=y_pred)

#model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
#
#mp = 'weights_densenet-19-2.30.h5'
mp = 'weights_densenet-02-1.29.h5'
modelPath = os.path.join(os.getcwd(), './models/' + mp)
if os.path.exists(modelPath):
	multi_model = multi_gpu_model(basemodel, gpus=GPU_NUM)
	#multi_model = basemodel
	multi_model.load_weights(modelPath)
	basemodel = multi_model
	#model.load_weights(modelPath)

print (basemodel.layers[-1])


def ctc_decode(pred):
	c = K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=False, beam_width=10)[0][0]
	print (c)

def decode(pred):
	char_list = []
	pred_text = pred.argmax(axis=2)[0]
	for i in range(len(pred_text)):
		if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
		#if pred_text[i] != nclass - 1:
			char_list.append(char_set[pred_text[i]])
	return u''.join(char_list)

def predict(img):
	width, height = img.size[0], img.size[1]
	scale = height * 1.0 / 32
	width = int(width / scale)
	print (width, height) 
	#width = 280
	img = img.resize([width, 32], Image.ANTIALIAS)
	print (img)
	'''
	img_array = np.array(img.convert('1'))
	boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
	if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
		img = ImageOps.invert(img)
	'''

	img = np.array(img).astype(np.float32) / 255.0 - 0.5
	print (img.shape)
	X = img.reshape([1, 32, width, 1])
	X = X.swapaxes(1,2)
	print("X", X.shape)
	y_pred = basemodel.predict(X)
	#print (y_pred.shape)
	print (y_pred[0])
	y_pred = y_pred[:, :, :]
	print (y_pred.argmax(axis=2)[0])
	# out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
	# out = u''.join([characters[x] for x in out[0]])
	out = decode(y_pred)

	#out = ctc_decode(y_pred)
	return out


if __name__ == '__main__':
	import sys
	input_image_path = sys.argv[1]
	if "jpg" in input_image_path or 'png' in input_image_path:
		img = Image.open(input_image_path).convert('L')
		print (predict(img))

	else:
		for i in range(20):
			img = Image.open(input_image_path + "/" + str(i) + ".jpg").convert('L')
			print (predict(img))
