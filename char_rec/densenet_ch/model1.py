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
from . import keys
#from . import densenet
#from . import crnn as densenet
from densenet_ch.eng_dict import eng_dict


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
    char_set = open('./densenet_ch/japeng.txt', 'r', encoding='utf-8').readlines()
else:
    char_set = open('./densenet_ch/chn.txt', 'r', encoding='utf-8').readlines()
for i in range (0, len(char_set)):
	c = char_set[i].strip('\n')
	encode_dct[c] = i
eng_dict = eng_dict('./densenet_ch/corpus/engset.txt')
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
    modelPath = os.path.join(os.getcwd(), './models/weights_eng_300_avg_2+3+4.h5')
else:
    if MODEL == 'resnet':
        # print('中文模型。。。。'#weights_chn_add_lishu_avg3+4+5.h5)#weights_eng_add_2_class_1_avg_3+4+5.h5
        modelPath = os.path.join(os.getcwd(), './models/weights_chn_add_lishu_test_2_avg_3_4.h5')#1+2+3.h5
    else:
        modelPath = os.path.join(os.getcwd(), './models/new_model_crnn.h5')
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
    second_score_list = []
    pred_text = pred.argmax(axis=2)[0]
    max_score_index = []
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1]))):
            max_score = max(pred[0][i])
            pred[0][i][pred_text[i]] = 0
            second_score_index = pred[0][i].argmax(axis = 0)
            #if char_set[pred_text[i]] == ' 'and max_score < 0.7:
            #    continue
            #print(pred_text[i],second_score_index)
            max_score_index.append(pred_text[i])
            second_score_list.append(second_score_index)
            char_list.append(char_set[pred_text[i]])
    
            score_list.append(max_score)
    #print(max_score_index,second_score_list)
    text = u''.join(char_list)
    before_change = text
    #print('修正前：',text)
    
    text_filter = ' '.join(list(filter(None,re.compile('[^A-Za-z^~]').sub(' ',text).split(' '))))
    word_list = text_filter.split(' ')
    text_list = list(text)
    #word_list = filter(None,word_list)
    word_dict = eng_dict.word_dict
    change_or_not = False
    for word in word_list:
        word_low = word.lower()
        if word_low in word_dict:
            #print('word in word_dict :',word)
            continue
        else:
            #print('word not in word_dict :',word,'text:',text)
            try:
                if ('('in word ) or (')' in word):
                    continue
            #if re.finditer(word, text):
            #    print('aaaaaaaaaa')
                for i in re.finditer(word, text):
                    #print('word,,,,,,')
                   
                    #print(score_list[i.start():i.end()],word)
                    up_count = text[:i.start()].count('^')+text[:i.start()].count('~')
                    wrong_w_index_list = np.where(np.array(score_list[i.start()-up_count:i.end()-up_count])<0.9)[0]
                    #print(word,wrong_w_index_list)
                    if len(wrong_w_index_list) < 3:
                        for n in wrong_w_index_list:
                            wrong_w_index = i.start() - up_count + n
                            if (second_score_list[wrong_w_index]!= nclass - 1) and word[wrong_w_index_list[0]]!= '/' :
                                text_list[wrong_w_index] = char_set[second_score_list[wrong_w_index]]
                                change_or_not = True       
                    else:
                        break
                continue
            except Exception as e:
                print(str(e))

                continue
    #text = text.replace('mgl','mg/')
    #return re.compile(u'[\u4E00-\u9FA5]').sub('',text), score_list
    text = ''.join(text_list)
    if change_or_not:
        print('修正前：',before_change)
        print('修正后：',text)
    #re.compile(u'[\u4E00-\u9FA5]').sub('',text)
    text = text.replace('^oC','°C')
    text = text.replace('mgl','mg/')
    #print(score_list)
    return text,score_list


def decode_ori(pred):
    char_list = []
    score_list = []
    pred_text = pred.argmax(axis=2)[0]
    max_score_index_list = []
    for i in range(len(pred_text)):
        #if pred_text[i] != nclass - 1: #and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1]))):
            max_score = max(pred[0][i])
            #if max_score < 0.1:  # 去掉概率小于0.1
             
            #    continue
            
            #try:
            #    if char_list[-1] == 'g' and char_list[-2] == 'm'and char_set[pred_text[i]] == 'l':
            #        char_set[pred_text[i]] = '/'
            #except:
            #    print('ssss')
            char_list.append(char_set[pred_text[i]])
            max_score_index_list.append(pred_text[i])
            score_list.append(max_score)
    text = u''.join(char_list)
    #text = text.replace('mgl','mg/')
    #return re.compile(u'[\u4E00-\u9FA5]').sub('',text), score_list
    #print(max_score_index_list)
    return text,score_list

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