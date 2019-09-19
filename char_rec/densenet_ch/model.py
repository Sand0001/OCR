#-*- coding:utf-8 -*-
import os
import numpy as np
from imp import reload
from PIL import Image, ImageOps
import cv2
import json
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
encode_dct =  {}
lfreq = json.loads(open('./char_rec/densenet_ch/count_big.json','r').readlines()[0])
char_set = open('./char_rec/densenet_ch/chn.txt', 'r', encoding='utf-8').readlines()
for i in range (0, len(char_set)):
	c = char_set[i].strip('\n')
	encode_dct[c] = i
eng_dict = eng_dict('./char_rec/densenet_ch/corpus/engset.txt')
char_set = [c.strip('\n') for c in char_set]
char_set.append('卍')

nclass = len(char_set)
punctuation = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
Easily_confused = ['入','人','血', '皿', '真', '直', '淋', '沛']
#Easily_confused_word = {'径':{'真径':'直径'},'入':{'传入':'传入'}}
Easily_confused_word = {'径':{'真径':'直径'}}
mult_model, basemodel = densenet.get_model(False, 32, nclass)
modelPath = os.path.join(os.getcwd(), './char_rec/models/weights_chn_filter_resnet-10one.h5')#weights_eng_finetune_300_finally_resnet-01-1.11.h5
if os.path.exists(modelPath):
    basemodel.load_weights(modelPath)
else:
    print ("No Chn Model Loaded, Default Model will be applied")
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
k = 0.00000001
def get_bigram_score(s):
    score = 1
    spilt_list = list(s)
    for i,word  in enumerate(spilt_list):
        if i == 0:
            word_combine = word
            score = 1*1
        else:
            word_combine = spilt_list[i-1]+word

        if word_combine in lfreq and i!=0:
            score = score * lfreq[word_combine]*1.0/lfreq[word]
            #print(word_combine,lfreq[word_combine])
        else:
            score = score* k
            #print(word_combine,0)
    return score
def char_in_Easily_confused_word(s1,s2):
    if s1 + s2 in Easily_confused_word[s2]:
        return Easily_confused_word[s2][s1 + s2]
def Vierbi_simple(text_tmp,score_list_tmp,second_score_list_index_tmp,thresh_tmp):
    thresh_big = 0.95
    thresh = 0.6
    wrong_w_index_list = []
    start_index = 0
    if len(thresh_tmp) >0:
        for j in thresh_tmp:
            wrong_w_index_list += np.where(np.array(score_list_tmp[start_index:j]) < thresh)[0].tolist()
            if score_list_tmp[j] < thresh_big:
                wrong_w_index_list += [j]
            if j!= len(text_tmp)-1:
                start_index = j+1
            else:
                break
    else:
        wrong_w_index_list = np.where(np.array(score_list_tmp) < thresh)[0]  # 查看临时 score_list中有没有低于0.9的
    if len(score_list_tmp) > 1 and len(wrong_w_index_list) <4:    #现在将限定一个字数解除，可更改多个错字

        for j in wrong_w_index_list:
            tmp_char_list = list(text_tmp)

            second_score_index = second_score_list_index_tmp[j].argmax(axis=0)  # second index
            second_score = second_score_list_index_tmp[j][second_score_index]
            tmp_char_list[j] = char_set[second_score_index]
            if j ==0:
                gama = 0.1  #惩罚因子
            else:
                gama = 1
            # s_score = get_bigram_score(text_tmp)
            # sb_score = get_bigram_score(''.join(tmp_char_list))   # 获取 校正后bigram分数
            s_score = get_bigram_score(text_tmp) ** gama*score_list_tmp[j]
            sb_score = get_bigram_score(''.join(tmp_char_list)) ** gama *second_score # 获取 校正后bigram分数
            if s_score > sb_score:
                text_tmp = text_tmp
            else:
                #print('转换前：',text_tmp)
                #print('转换后: ',''.join(tmp_char_list))
                text_tmp = ''.join(tmp_char_list)

    #else:
    text_tmp_final = text_tmp  # 将临时text 合并到最终返回的text里
    return text_tmp_final
def decode_Viterbi(pred):
    pred_text = pred.argmax(axis=1)
    text_final = ''
    text_tmp = ''
    score_list_tmp = []
    second_score_list_index_tmp = []
    score_list = []
    thresh_tmp = []
    num_symble = 0
    for i in range(len(pred_text)):
        # pred if pred_text[i] != nclass - 1: #and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1]))):
            max_score = pred[i][pred_text[i]]
            pred[i][pred_text[i]] = 0
            second_score_list_index_tmp.append(pred[i])

            char = char_set[pred_text[i]]
            if char not in punctuation:
                score_list_tmp.append(max_score)
                if char in Easily_confused_word:  # 向前寻找词语，向前查找一个字 强制替换易混词  现在先查找一个字 ，之后根据统计添加查找多个字
                    if len(text_tmp) > 0:
                        char_word = char_in_Easily_confused_word(text_tmp[-1], char)
                        if char_word:
                            text_tmp = text_tmp[:-1] + char_word
                        else:
                            text_tmp += char
                    else:
                        text_tmp += char
                elif char in Easily_confused:
                    # thresh_tmp.append(True)  # 判断阈值的tmp_list
                    thresh_tmp.append(num_symble)
                    text_tmp += char
                else:
                    # thresh_tmp.append(False)
                    text_tmp += char
                num_symble += 1
            else:
                text_final += Vierbi_simple(text_tmp, score_list_tmp, second_score_list_index_tmp, thresh_tmp)
                # text_final += text_tmp_final
                text_final += char  # 将else的标点也加上
                score_list += score_list_tmp  #
                score_list.append(max_score)  # 目前分数列表里不管有没有矫正存的都是分数最大值

                second_score_list_index_tmp = []  # 将临时 second score index list 初始化
                score_list_tmp = []  # 将临时最大分数list初始化
                thresh_tmp = []
                num_symble = 0
                text_tmp = ''  # 将临时text 初始化
                continue
    if len(text_tmp) > 1:  # 为防止遗漏 ，还是加一下text_tmp
        text_tmp_final = Vierbi_simple(text_tmp, score_list_tmp, second_score_list_index_tmp, thresh_tmp)
        # text_tmp_final = text_tmp
    else:
        text_tmp_final = text_tmp

    text_final += text_tmp_final
    score_list += score_list_tmp
    strQ2B_text = strQ2B(text_final, score_list)
    # if len(text_final) > 0:
    #     if text_final[-1] == ']' or strQ2B_text[-1] == '卍':
    #         text_final = text_final[:-1]
    #         score_list = score_list[:-1]
    score_list = [str(ele) for ele in score_list]
    #strQ2B_text = strQ2B_text.
    strQ2B_text = strQ2B_text.replace('▿',' ')
    strQ2B_text = strQ2B_text.replace('▵','　')
    return strQ2B_text, score_list
def predict_batch(img,image_info):
    y_pred = basemodel.predict_on_batch(img)[:,2:,:]
    result_info = []
    #logging.info('chn batch')
    for i in range(len(y_pred)):
        text,scores = decode_Viterbi(y_pred[i])
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
