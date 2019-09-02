#-*- coding:utf-8 -*-
import os
import numpy as np
from imp import reload
from PIL import Image, ImageOps

from keras.layers import Input
from keras.models import Model
import keras.backend as K
from keras.utils import multi_gpu_model

from . import keys
#from . import densenet
#from . import crnn as densenet



#reload(densenet)

LAN = 'chn'
LAN = 'jap'
MODEL = 'resnet'
#MODEL = 'crnn'

if MODEL == 'resnet' and LAN == 'jap':
    from . import dl_resnet_crnn_cudnnlstm as densenet

#LAN = 'jap'
'''
characters = keys.alphabet[:]
characters = characters[1:] + u' '
characters = ''.join([chr(i) for i in range(32, 127)] + ['卍'])
nclass = len(characters)
nclass = 96
input = Input(shape=(32, None, 1), name='the_input')
y_pred= densenet.dense_cnn(input, nclass)
basemodel = Model(inputs=input, outputs=y_pred)
'''
encode_dct =  {}

if LAN == 'jap':
    char_set = open('./char_rec/densenet_jp/japeng.txt', 'r', encoding='utf-8').readlines()
else:
    char_set = open('./char_rec/densenet_jp/chn.txt', 'r', encoding='utf-8').readlines()
for i in range (0, len(char_set)):
	c = char_set[i].strip('\n')
	encode_dct[c] = i

char_set = [c.strip('\n') for c in char_set]
char_set.append('卍')
#r_char_set = ''
#for c in char_set:
#    r_char_set += c
#    #char_set = ''.join([ch.strip('\n') for ch in char_set] + ['卍'])
#r_char_set += '卍'
#char_set = r_char_set
#print (char_set[encode_dct[" "]])
#print (char_set[encode_dct["语"]])


nclass = len(char_set)
# print(nclass)
mult_model, basemodel = densenet.get_model(False, 32, nclass)
if LAN == 'jap':
    # print('日文模型。。。')
    modelPath = os.path.join(os.getcwd(), './char_rec/model_new/avg2+3+4_big_japeng.h5')
else:
    if MODEL == 'resnet':
        modelPath = os.path.join(os.getcwd(), './char_rec/models/chn_eng_20190617.h5')
    else:
        modelPath = os.path.join(os.getcwd(), './char_rec/models/new_model_crnn.h5')
if os.path.exists(modelPath):
    #multi_model = multi_gpu_model(basemodel, 4, cpu_relocation=True)
    #multi_model.load_weights(modelPath)
    #basemodel = multi_model
    basemodel.load_weights(modelPath)
else:
    print ("No Model Loaded, Default Model will be applied")
    import sys
    sys.exit(-1)
def decode(pred):
    char_list = []
    score_list = []
    pred_text = pred.argmax(axis=1)
    for i in range(len(pred_text)):
        #if pred_text[i] != nclass - 1: #and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1]))):
            max_score = max(pred[i])
            if max_score < 0.1:  # 去掉概率小于0.1
                continue 
            char_list.append(char_set[pred_text[i]])
            score_list.append(str(max_score))
    return u''.join(char_list), score_list
def predict_batch(img,image_info):
    y_pred = basemodel.predict_on_batch(img)
    result_info = []
    for i in range(len(y_pred)):
        text,scores = decode(y_pred[i])
        scores = [float(ele) for ele in scores]
        #rec = rec.tolist()
        #rec.append(degree)
        if len(text) > 0:
            if len(scores) == 1:
                imagename = {}
                imagename['location'] = image_info[i]['location']
                imagename['text'] = text
                imagename['scores'] = [str(ele) for ele in scores]
                result_info.append(imagename)
            elif len(scores)>1 and (sum(scores)*1.0/len(scores)>0.6):
                imagename = {}
                imagename['location'] = image_info[i]['location']
                imagename['text'] = text
                imagename['scores'] = [str(ele) for ele in scores]
                result_info.append(imagename)
    return result_info
def predict(img):
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)
    
    img = img.resize([width, 32], Image.ANTIALIAS)
   
    '''
    img_array = np.array(img.convert('1'))
    boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
    if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
        img = ImageOps.invert(img)
    '''

    img = np.array(img).astype(np.float32) / 255.0 - 0.5
    
    X = img.reshape([1, 32, width, 1])
    X = X.swapaxes(1,2)
    y_pred = basemodel.predict(X)
    y_pred = y_pred[:, 2:, :]
   # outTxt = open('out.txt','w')
   # for i in range(len(y_pred[0]))
        
    #out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1],greedy=False, beam_width=100, top_paths=2)[0][0])[:, :]
    #out = u''.join([characters[x] for x in out[0]])
    #print(y_pred.shape)
    out = decode(y_pred)
    
    return out
