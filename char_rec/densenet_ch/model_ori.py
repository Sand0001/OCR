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
from . import keys
#from . import densenet
#from . import crnn as densenet
from char_rec.densenet_ch.eng_dict import eng_dict
import time

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
    char_set = open('./char_rec/densenet_ch/japeng.txt', 'r', encoding='utf-8').readlines()
else:
    char_set = open('./char_rec/densenet_ch/chn7231.txt', 'r', encoding='utf-8').readlines()
for i in range (0, len(char_set)):
	c = char_set[i].strip('\n')
	encode_dct[c] = i
eng_dict = eng_dict('./char_rec/densenet_ch/corpus/engset.txt')
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
    modelPath = os.path.join(os.getcwd(), './char_rec/models/avg2+3+4_big_japeng.h5')
else:
    if MODEL == 'resnet':
        # print('中文模型。。。。')
        modelPath = os.path.join(os.getcwd(), './char_rec/model_new/weights_chn_new_819_avg2+3+4.h5')#weights_eng_finetune_300_finally_resnet-01-1.11.h5
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
def is_chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def del_blank(word):
    word = list(filter(None, word.strip().split(' ')))
    if len(word)>0:
        c = word[0]
    else:
        return ''
    for i in range(len(word) - 1):
        if not ('\u4e00' <= word[i][-1] <= '\u9fff'):
            if ord(word[i][-1]) > 47 and ord(word[i][-1]) <= 57:
                if len(re.compile(r'\b[a-zA-Z]+\b', re.I).findall(word[i + 1][0])) != 0:
                    c = c + ' ' + word[i + 1]
                else:
                    c = c + word[i + 1]
            elif len(re.compile(r'\b[a-zA-Z]+\b', re.I).findall(word[i][-1])) == 0:
                if len(re.compile(r'\b[a-zA-Z]+\b', re.I).findall(word[i + 1][0])) != 0:
                    c = c + ' ' + word[i + 1]
                else:
                    c = c + word[i + 1]
            elif len(re.compile(r'\b[a-zA-Z]+\b', re.I).findall(word[i][-1])) != 0:
                if '\u4e00' <= word[i+1][0] <= '\u9fff':
                    c = c + word[i + 1]
                else:
                    c = c + ' ' + word[i + 1]
            else:
                if '\u4e00' <= word[i+1][0] <= '\u9fff':
                    c = c + word[i + 1]
                elif len(re.compile(r'\b[a-zA-Z]+\b',re.I).findall(word[i+1][0]))!= 0:
                        c = c + ' ' + word[i + 1]
                else:
                    c = c + word[i + 1]
        else:
            c = c  + word[i + 1]
    return c

def strQ2B(a, max_score_list):
    if is_chinese(a):
        a = a.replace('(','（')
        a = a.replace(')','）')
        a = a.replace(',','，')
        a = a.replace(':', '：')
        a = a.replace('?', '？')
        
        list_a = list(a)
        for i in re.finditer('\.|\(|（|）|\)', a):
            if list_a[i.start()] == '.':
                if max_score_list[i.start()] <0.9:
                    if re.compile(u'[\u4E00-\u9FA5]').findall(a[i.start()-1]):  #判定中文
                        list_a[i.start()] ='。'
                    elif a[i.start()-1] ==')' and ((ord(a[i.start()-2]) > 47 and ord(a[i.start()-2]) < 59) or re.compile(u'[\u4E00-\u9FA5]').findall(a[i.start()-2])):
                        list_a[i.start()] = '。'
                    elif a[i.start()-1] =='）':
                        list_a[i.start()] = '。'
                    elif a[i.start()-1] =='）':
                        list_a[i.start()] = '。'
        a = ''.join(list_a)
        a = del_blank(a)
        return a
    else:
        a = a.replace('（', '(')
        a = a.replace('）', ')')
        #a = a.replace('。','.')
        a = a.replace('：', ':')
        a = a.replace('？', '?')
        a = del_blank(a)
        return a

def strQ2B_oold(a,max_score_list):
    if is_chinese(a):
        a = a.replace('(','（')
        a = a.replace(')','）')
        list_a = list(a)
        for i in re.finditer('\.|\(|（|）|\)', a):
            if list_a[i.start()] == '.':
                if max_score_list[i.start()] <0.9:
                    if re.compile(u'[\u4E00-\u9FA5]').findall(a[i.start()-1]):  #判定中文
  
                        list_a[i.start()] ='。'
                    elif a[i.start()-1] ==')' and ((ord(a[i.start()-2]) > 47 and ord(a[i.start()-2]) < 59) or re.compile(u'[\u4E00-\u9FA5]').findall(a[i.start()-2])):
                        list_a[i.start()] = '。'
                    elif a[i.start()-1] =='）':
                        list_a[i.start()] = '。'
                    elif a[i.start()-1] =='）':
                        list_a[i.start()] = '。'
        return ''.join(list_a)
    else:
        a = a.replace('（', '(')
        a = a.replace('）', ')')
    return a

def decode_new(pred):
    char_list = []
    score_list = []
    second_score_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        #if pred_text[i] != nclass - 1: #and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1]))):
            max_score = max(pred[0][i])
            pred[0][i][pred_text[i]] = 0
            second_score_index = pred[0][i].argmax(axis = 0)
            #print(pred_text[i],second_score_index)
            second_score_list.append(second_score_index)
            #if max_score < 0.1:  # 去掉概率小于0.1

            #    continue

            #try:
            #    if char_list[-1] == 'g' and char_list[-2] == 'm'and char_set[pred_text[i]] == 'l':
            #        char_set[pred_text[i]] = '/'
            #except:
            #    print('ssss')
            char_list.append(char_set[pred_text[i]])
            score_list.append(float(max_score))

    text = u''.join(char_list)
    word_list = text.split(' ')
    text_list = list(text)
    word_list = filter(None,word_list)
    word_dict = eng_dict.word_dict
    for word in word_list:
        if word in word_dict:
            #print('word in word_dict',word)
            continue
        else:
            #print('word not in word_dict',word,'text:',text)
            try:

                for i in re.finditer(word, text):
                    wrong_w_index_list = np.where(np.array(score_list[i.start():i.end()])<0.9)[0]
                    if len(wrong_w_index_list) < 3:
                        for n in wrong_w_index_list:
                            wrong_w_index = i.start() + wrong_w_index_list[0]
                  
                            if second_score_list[wrong_w_index]!= nclass - 1:

                                text_list[wrong_w_index] = char_set[second_score_list[wrong_w_index]]
                    else:
                        break
                continue
            except:
                continue
    #text = text.replace('mgl','mg/')
    #return re.compile(u'[\u4E00-\u9FA5]').sub('',text), score_list
    text = ''.join(text_list)
    #re.compile(u'[\u4E00-\u9FA5]').sub('',text)
    text = text.replace('^oC','°C')
    return text,score_list


def decode(pred):
    char_list = []
    score_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        #if pred_text[i] != nclass - 1: #and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1]))):
            max_score = max(pred[0][i])
            if max_score < 0.1:  # 去掉概率小于0.1
             
                continue
            
            #try:
            #    if char_list[-1] == 'g' and char_list[-2] == 'm'and char_set[pred_text[i]] == 'l':
            #        char_set[pred_text[i]] = '/'
            #except:
            #    print('ssss')
            char_list.append(char_set[pred_text[i]])
            score_list.append(str(max_score))
    text = u''.join(char_list)
    #text = text.replace('mgl','mg/')
    #return re.compile(u'[\u4E00-\u9FA5]').sub('',text), score_list
    float_score_list = [float(i) for i in score_list]
    text = strQ2B(text,float_score_list)
    return text,score_list
def strQ2B_old(a):
    list_a = list(a)
    for i in re.finditer('\.', a):
        if re.compile(u'[\u4E00-\u9FA5]').findall(a[i.start()-1]):  #判定中文

            list_a[i.start()] ='。'
        elif a[i.start()-1] ==')' and ((ord(a[i.start()-2]) > 47 and ord(a[i.start()-2]) < 59) or re.compile(u'[\u4E00-\u9FA5]').findall(a[i.start()-2])):
        #elif a[i.start()-1] ==')' and (ord(a[i.start()-2]) > 47 and ord(a[i.start()-2]) < 59):
            list_a[i.start()] = '。'
        elif a[i.start()-1] =='）':
            list_a[i.start()] = '。'
    return ''.join(list_a)
def predict(img):
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)
    img = np.array(img)
    img = cv2.resize(img,(width,32),interpolation=cv2.INTER_AREA)
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
    t1 = time.time()
    y_pred = basemodel.predict(X)
    t_pred = time.time()-t1
    # print('单行预测时间',t_pred)
    # logging.info('单行预测时间:%s',str(t_pred))
    y_pred = y_pred[:, 2:, :]
   # outTxt = open('out.txt','w')
   # for i in range(len(y_pred[0]))
        
    #out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1],greedy=False, beam_width=100, top_paths=2)[0][0])[:, :]
    #out = u''.join([characters[x] for x in out[0]])
    #print(y_pred.shape)
    t2 = time.time()
    out = decode(y_pred)
    t_decode = time.time() - t2
    # print("单行decode时间",t_decode)
    # logging.info('单行decode时间:%s',t_decode)
    return out
