#-*- coding:utf-8 -*-
import os
import numpy as np
from imp import reload
from PIL import Image, ImageOps
import itertools
import json

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
from . import dl_resnet_crnn_cudnnlstm as densenet

#LAN = 'jap'
encode_dct =  {}
lfreq_word = json.loads(open('./char_rec/densenet_ch/count_word_chn0.json','r').readlines()[0])
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
modelPath = os.path.join(os.getcwd(), './char_rec/models/weights_eng_902_change_symbol_ratio_avg16+17+18.h5')#weights_eng_700_avg_1+2+3.h5
if os.path.exists(modelPath):
    #multi_model = multi_gpu_model(basemodel, 4, cpu_relocation=True)
    #multi_model.load_weights(modelPath)
    #basemodel = multi_model
    basemodel.load_weights(modelPath)
else:
    print ("No eng Model Loaded, Default Model will be applied")
    import sys
    sys.exit(-1)
def isalpha(c):
    if c <= 'z' and c >= 'a' or c >= 'A' and c <= 'Z':
        return True
    # if c <= '9' and c >= '0':
    #     return True
    # if c == '.' or c == ',':
    #     return True

    return False
def isnum(c):
    if ord(c) > 47 and ord(c) <= 57:
        return True
    return False
def is_chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False
def filter_blank(word,score_list):
    if len(word) - word.count('▿') - word.count('▵') != len(score_list):
        return word,score_list
    else:

        word_list = list(word)
        #word_tmp_list = word.split(' ')
        #space_num = 0
        for index ,w in enumerate(word):
            if w ==' ':

                if index == 0 or index == len(word) -1:# 判断句首和句尾
                    word_list[index] = ''
                    score_list[index] = ''
                else:

                    if is_chinese(word[index - 1]) or is_chinese(word[index + 1]):  # 如果前后出现汉字 空格去掉
                        word_list[index] = ''
                        score_list[index] = ''
                    elif word[index + 1] == ' ':  # 如果后面是空格，则 空格去掉
                        word_list[index + 1] = ''
                        score_list[index + 1] = ''
                    # elif word[index +1] == 'A' or word[index +1] == 'C' or word[index -1] == 'A' or word[index -1] == 'C'  :
                    #     if word_tmp_list[space_num -1] +word_tmp_list[space_num+1].lower() in word_dict:
                    #         if score_list[index] < 0.8:
                    #             word_list[index] = ''
                    #             score_list[index] = ''
                #space_num +=1



        #print(''.join(filter(None,word_list)))
        text = ''.join(list(filter(None,word_list)))
        if len(text) ==1 and text[0] == ' ':
            return '',[]

        return text,list(filter(None,score_list))
k = 0.00000001
def get_word_bigram_score(word_list):#没考虑开头和结尾
    score = 1
    for i, word in enumerate(word_list):
        if word != ' ':
            word_combine = ''
            if i == 0:
                word_combine = word
                score = 1 * 1
            else:
                for j in range(1,i+1):
                    if word_list[i - j] != ' ' and word_list[i - j] != '卍':
                        word_combine = word_list[i - j]+' ' + word     # 这一步是要得到当前词与前面不是空格的词的组合
                        break
            #print(word_combine)
            if word_combine in lfreq_word and i!=0 and word in lfreq_word :
                score = score * lfreq_word[word_combine]*1.0/lfreq_word[word]
                #print(word_combine,lfreq[word_combine])
            else:
                score = score* k
    return score
def eng_error_correction(text_tmp_list,score_list_tmp,wrong_charindex_list,text_tmp):
    tmp_word_list = []
    tmp_dict = {}
    if len(wrong_charindex_list) ==0:
        score = get_list_score(score_list_tmp)
        return [{text_tmp:{'score':score,'score_list':score_list_tmp}}]
    for i in itertools.product(*text_tmp_list):
        word = ''.join(list(i))
        #if '卍' in word:  # 如果占位符在还得减去一个score 麻烦
        score = 1
        if word.replace('卍','').lower() in word_dict:   #将占位符过滤掉
            score_list = []
            for index, w in enumerate(word):
                score *= text_tmp_list[index][w]['score']
                if w != '卍':          # 将占位符的概率也加进去计算
                    score_list.append(text_tmp_list[index][w]['score'])

            tmp_dict[word.replace('卍','')] ={'score':score,'score_list':score_list}
    if tmp_dict == {}:
        tmp_dict[text_tmp] = {'score':get_list_score(score_list_tmp),'score_list':score_list_tmp}

    tmp_word_list.append(tmp_dict)
    if len(tmp_word_list)>0:
        # print('转换前',text_tmp)
        # print('转换后',tmp_word_list)
        return tmp_word_list
    else:
        score = 1
        for s in score_list_tmp:
            score*= s
        return [{text_tmp:{'score':score,'score_list':score_list_tmp}}]
def get_list_score(score_list):
    score = 1
    for i in score_list:
        score*= i
    return score
def decode_eng_1(pred):
    pred_text = pred.argmax(axis=1)
    text_tmp = '' #存放临时单词
    text_tmp_list = []  #存放临时的字符 + score
    score_list_tmp = []  #存放tmp Max score
    wrong_charindex_list = []  #存放tmp 嫌疑字符列表
    wrong_charindex = 0
    word_list = [] #存放word
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1]))):
            max_score = pred[i][pred_text[i]]
            pred[i][pred_text[i]] = 0
            char = char_set[pred_text[i]]
            if isalpha(char):
                if max_score < 0.9:
                    wrong_charindex_list.append(wrong_charindex)  #嫌疑字index列表
                    second_char_index = pred[i].argmax(axis = 0)
                    text_tmp_list.append({char:{'score':max_score},char_set[second_char_index]:{'score':pred[i][second_char_index]}})
                else:
                    text_tmp_list.append({char:{'score':max_score}})
                text_tmp+=char
                score_list_tmp.append(max_score)
                wrong_charindex += 1
            else:
                if max_score <0.9:  # 优化字符识别成特殊字符的情况
                    second_char_index = pred[i].argmax(axis=0)
                    second_char = char_set[second_char_index]
                    if isalpha(second_char):
                        text_tmp_list.append({char:{'score':max_score} , second_char:{'score':pred[i][second_char_index]} })
                        continue
                if len(text_tmp_list) >1:
                    if text_tmp.lower() in word_dict:    #如果是单词 继续
                        score_list_tmp =score_list_tmp
                        text_tmp_1 = [{text_tmp:{'score':get_list_score(score_list_tmp),'score_list':score_list_tmp}}]

                        need_bigram = False
                    else: #如果不是单词 纠错
                        text_tmp_1 = eng_error_correction(text_tmp_list, score_list_tmp, wrong_charindex_list,text_tmp)  #得到修正后的单词列表
                    word_list +=(text_tmp_1)
                    text_tmp = ''
                    wrong_charindex = 0
                    wrong_charindex_list = []
                    score_list_tmp = []
                    text_tmp_list = []
                elif len(text_tmp_list) == 1:
                    word_list+= text_tmp_list
                    text_tmp = ''
                    wrong_charindex = 0
                    wrong_charindex_list = []
                    score_list_tmp = []
                    text_tmp_list = []
                #if is_chinese(char): #对汉字的处理

                if  max_score < 0.6 and ('▵'  not in char) and ('▿' not in char):  #如果字符是易混淆字符且概率小于0。95 或者最大值小于0。6

                    second_char_index = pred[i].argmax(axis=0)   # 这里备用字符不做是否占位符的判断
                    #if second_char_index != nclass - 1:
                    second_char = char_set[second_char_index]
                    char  = {char:{'score':max_score},second_char:{'score':pred[i][second_char_index]}}
                    word_list.append(char)
                else:
                    #print('else',{char:{'score':max_score}})
                    word_list.append({char:{'score':max_score}})
    if len(text_tmp_list)>1:
        text_tmp_1 = eng_error_correction(text_tmp_list, score_list_tmp, wrong_charindex_list,text_tmp)
        word_list+= text_tmp_1
    else:
        word_list += text_tmp_list
    word_list = list(filter(None, word_list))  #必须要过滤 否则path为空
    paths = list(itertools.product(*word_list))
    word_bigram_score_list = []
    score_list_final = []
    if len(paths)> 1:
        #print('rrrrr')
        gamma = 1  #发射概率的阈值
        alpha = 0.5  #LM 的阈值
        for path in paths:
            word_bigram_score_path = []
            word_bigram_score = get_word_bigram_score(path)**alpha
            path= list(path)
            p_pred = 1
            for j in range(len(path)):
                #try:
                score_path = word_list[j][path[j]]['score']  #获得每个词的分数
                p_pred *= score_path
                #except:
                #    print('score_path',word_list)
                if 'score_list' in  word_list[j][path[j]]:
                    word_bigram_score_path+=word_list[j][path[j]]['score_list']
                else:
                    if path[j] != '卍':
                        word_bigram_score_path += [word_list[j][path[j]]['score']]   #获得每个path中每个字符分数列表
            word_bigram_score_list.append(word_bigram_score*p_pred**gamma) if len(path) > 2 else word_bigram_score_list.append(p_pred)   #
            score_list_final.append(word_bigram_score_path)
        max_score_index = np.argmax(np.array(word_bigram_score_list),axis=0)
        #print(''.join(paths[max_score_index]))
        final_score = score_list_final[max_score_index]
        final_text = ''.join(paths[max_score_index]).replace('卍','')
        #assert len(final_text)-final_text.count('▵') - final_text.count('▿') == len(final_score)
        #final_text,final_score = filter_blank(final_text,final_score)
        strQ2B_text = final_text.replace('▿', ' ')
        strQ2B_text = strQ2B_text.replace('▵', '　')
        return strQ2B_text,final_score
        #return None,None
    elif len(paths)==1 :
        #score_list_final = []
        path = list(paths[0])
        for j in range(len(path)):
            if 'score_list' in word_list[j][path[j]]:
                #try:
                score_list_final += word_list[j][path[j]]['score_list']
            else:
                #try:
                score_list_final.append(word_list[j][path[j]]['score'])
                #except:
                #    print('word_list[j][path[j]]',word_list[j][path[j]])
        final_text = ''.join(list(paths[0]))
        #assert len(final_text)-final_text.count('▵') - final_text.count('▿') == len(score_list_final)
        #final_text,final_score = filter_blank(final_text,score_list_final)
        strQ2B_text = final_text.replace('▿', ' ')

        strQ2B_text = strQ2B_text.replace('▵', '　')
        return strQ2B_text,score_list_final   #,score_list     ###score  等下再拿出来
def predict_batch(img,image_info):
    y_pred = basemodel.predict_on_batch(img)[:,2:,:]
    result_info = []
    #logging.info('eng batch')
    for i in range(len(y_pred)):
        try:
            text,scores = decode_eng_1(y_pred[i])
        except:
            logging.info(e)
            text,scores = decode_ori_ori(y_pred[i])
        #if text != text_ori:
        len_scores = len(scores)
        len_text = len(text)
        if len_text > len_scores:
            for j in range(len_text - len_scores):
                scores.append(0.58)
        elif len_text <len_scores:
            scores = scores[:len_text]
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
