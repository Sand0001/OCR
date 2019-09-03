#-*- coding:utf-8 -*-
import os
import numpy as np
from imp import reload
from PIL import Image, ImageOps
import cv2
from keras.layers import Input
from keras.models import Model
import keras.backend as K
from keras.utils import multi_gpu_model
import re
import logging
import pickle
from . import keys
#from . import densenet
#from . import crnn as densenet
#from densenet_ch.eng_dict import eng_dict


#reload(densenet)

LAN = 'chn'
# LAN = 'jap'
MODEL = 'resnet'
#MODEL = 'crnn'

if MODEL == 'resnet' and LAN == 'JAP':
    from . import dl_resnet_crnn as densenet
else:
    from . import dl_resnet_crnn_cudnnlstm as densenet

#LAN = 'jap'
encode_dct =  {}

pkl_file = open('./char_rec/densenet_eng/eng_dict.pkl', 'rb')
word_dict = pickle.load(pkl_file)
char_set = open('./char_rec/densenet_eng/eng_new.txt', 'r', encoding='utf-8').readlines()
for i in range (0, len(char_set)):
	c = char_set[i].strip('\n')
	encode_dct[c] = i
#eng_dict = eng_dict('./densenet_ch/corpus/engset.txt')
char_set = [c.strip('\n') for c in char_set]
char_set.append('卍')

nclass = len(char_set)
# print(nclass)
mult_model, basemodel = densenet.get_model(False, 32, nclass)
modelPath = os.path.join(os.getcwd(), './char_rec/models/weights_eng_subscripts_823_test3_avg2+3+4.h5')#weights_eng_700_avg_1+2+3.h5
if os.path.exists(modelPath):
    #multi_model = multi_gpu_model(basemodel, 4, cpu_relocation=True)
    #multi_model.load_weights(modelPath)
    #basemodel = multi_model
    basemodel.load_weights(modelPath)
else:
    print ("No eng Model Loaded, Default Model will be applied")
    import sys
    sys.exit(-1)
def decode(pred):
    char_list = []
    score_list = []
    second_score_list = []
    pred_text = pred.argmax(axis=1)
    max_score_index = []
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1]))):
            max_score = max(pred[i])
            pred[i][pred_text[i]] = 0
            second_score_index = pred[i].argmax(axis = 0)
            #if char_set[pred_text[i]] == ' 'and max_score < 0.7:
            #    continue
            max_score_index.append(pred_text[i])
            second_score_list.append(second_score_index)
            char_list.append(char_set[pred_text[i]])

            score_list.append(max_score)
    text = u''.join(char_list)
    before_change = text

    text_filter = ' '.join(list(filter(None,re.compile('[^A-Za-z▵▿]').sub(' ',text).split(' '))))
    word_list = text_filter.split(' ')
    text_list = list(text)
    #word_dict = eng_dict.word_dict
    change_or_not = False
    for word in word_list:
        word_low = word.lower()
        if ('▿' or '▵') in word:
            continue
        if word_low in word_dict:
            continue
        else:
            try:
                if ('('in word ) or (')' in word):
                    continue
                for i in re.finditer(word, text):

                    wrong_w_index_list = np.where(np.array(score_list[i.start():i.end()])<0.9)[0]
                    if len(wrong_w_index_list) < 2:
                        for n in wrong_w_index_list:
                            word_list = list(word)
                            wrong_w_index = i.start() + n
                            if word[wrong_w_index_list[0]]!= '/' :
                                if second_score_list[wrong_w_index]== nclass - 1:
                                    word_list[n] = ''
                                else:
                                    word_list[n] = char_set[second_score_list[wrong_w_index]]
                                if ''.join(word_list).lower() in word_dict:
                                    text_list[wrong_w_index] = word_list[n]
                    else:
                        break
                continue
            except Exception as e:
                print(str(e))

                continue
    text = text.replace('▿',' ')
    text = text.replace('▵','　')
    return text,[str(i) for i in score_list]

def predict_batch(img,image_info):
    y_pred = basemodel.predict_on_batch(img)[:,2:,:]
    result_info = []
    #logging.info('eng batch')
    for i in range(len(y_pred)):
        text,scores = decode(y_pred[i])
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
    img = np.array(img)
    img = cv2.resize(img,(width,32),interpolation=cv2.INTER_CUBIC)
    #img = img.resize([width, 32], Image.ANTIALIAS)
   
    '''
    img_array = np.array(img.convert('1'))
    boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
    if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
        img = ImageOps.invert(img)
    '''

    #img = np.array(img).astype(np.float32) / 255.0 - 0.5
    img = img.astype(np.float32) / 255.0 - 0.5
    
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
