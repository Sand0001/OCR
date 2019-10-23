#-*- coding:utf-8 -*-
import os
import numpy as np
from imp import reload
from PIL import Image, ImageOps
import cv2
import json
import pickle
import itertools
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
#lfreq = json.loads(open('./char_rec/densenet_ch/count_big.json','r').readlines()[0])
lfreq_word = json.loads(open('./char_rec/densenet_ch/char_and_word_bigram_chneng.json','r').readlines()[0])
char_set = open('./char_rec/densenet_ch/chn.txt', 'r', encoding='utf-8').readlines()
for i in range (0, len(char_set)):
	c = char_set[i].strip('\n')
	encode_dct[c] = i
eng_dict = eng_dict('./char_rec/densenet_ch/corpus/engset.txt')
char_set = [c.strip('\n') for c in char_set]
char_set.append('卍')
nclass = len(char_set)

pkl_file = open('./char_rec/densenet_eng/eng_dict.pkl', 'rb')
word_dict = pickle.load(pkl_file)
punctuation = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
Easily_confused = ['入','人','血', '皿', '真', '直', '淋', '沛']
#Easily_confused_word = {'径':{'真径':'直径'},'入':{'传入':'传入'}}
Easily_confused_word = {'径':{'真径':'直径'}}
mult_model, basemodel = densenet.get_model(False, 32, nclass)
modelPath = os.path.join(os.getcwd(), './char_rec/models/weights_chn_0925_resnet-05-one.h5')#weights_eng_finetune_300_finally_resnet-01-1.11.h5
if os.path.exists(modelPath):
    basemodel.load_weights(modelPath)
else:
    print ("No Chn Model Loaded, Default Model will be applied")
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
        print('一开始就不相等')
        return word,score_list
    else:

        word_list = list(word)
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
        #print(''.join(filter(None,word_list)))
        text = ''.join(list(filter(None,word_list)))
        if len(text) ==1 and text[0] == ' ':
            return '',[]

        return text,list(filter(None,score_list))
def strQ2B(a,max_score_list):
    t4 = time.time()

    is_chinese_or = is_chinese(a)
    #print('判断是否是中文时间：',time.time() - t4)
    if is_chinese_or:
        a = a.replace('(','（')
        a = a.replace(')','）')
        a = a.replace(',','，')
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
        t3 = time.time()

        #a = filter_blank(a, max_score_list)
        #a = del_blank(a)
        #print("del blank时间为：",time.time() - t3)
        
    else:
        a = a.replace('（', '(')
        a = a.replace('）', ')')
        #a = a.replace('。','.')
        t3 = time.time()
        #a = filter_blank(a, max_score_list)
        #print("del blank时间为：",time.time() - t3)
    return a,max_score_list

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
            if word_combine in lfreq_word and i!=0 and word in lfreq_word:
                score = score * lfreq_word[word_combine]*1.0/lfreq_word[word]
                #print(word_combine,lfreq[word_combine])
            else:
                score = score* k
    return score

def char_in_Easily_confused_word(s1,s2):
    if s1 + s2 in Easily_confused_word[s2]:
        return Easily_confused_word[s2][s1 + s2]

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

def decode_chn_eng(pred):
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
                        # if text_tmp == 'Crossiand':
                        #     print('哈哈哈哈哈哈哈', text_tmp)
                        #print('text_tmp',text_tmp)
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
                if char in Easily_confused_word: #如果在易混淆词库
                    s1 = list(word_list[-1].keys())[0]
                    if len(word_list) > 0 and char_in_Easily_confused_word(s1, char):
                        tmp_s = char_in_Easily_confused_word(s1, char)

                        word_list[-1][tmp_s[0]] = word_list[-1][s1]
                        word_list[-1].pop(s1)  #将原字符删除掉
                        #text_tmp = text_tmp[:-1] + char_in_Easily_confused_word(text_tmp[-1], char)
                    word_list.append({char: {'score': max_score}})
                    #print('if',{char: {'scores': max_score}})

                elif (char in Easily_confused and max_score < 0.95) or (max_score < 0.6 and ('▵'  not in char) and ('▿' not in char)):  #如果字符是易混淆字符且概率小于0。95 或者最大值小于0。6

                    second_char_index = pred[i].argmax(axis=0)   # 这里备用字符不做是否占位符的判断
                    #if second_char_index != nclass - 1:
                    second_char = char_set[second_char_index]
                    char  = {char:{'score':max_score},second_char:{'score':pred[i][second_char_index]}}
                    word_list.append(char)
                else:
                    #print('else',{char:{'score':max_score}})
                    word_list.append({char:{'score':max_score}})
                '''
                if max_score<0.9:  #对符号的操作
                    second_char_index = pred[i].argmax(axis = 0)
                    if second_char_index != nclass - 1:
                        second_char = char_set[second_char_index]
                        #if
                        if ('°' != char) and ('▵'  not in char) and ('▿' not in char):
                            if '▵' in second_char or '▿' in second_char :
                                #print('获取捞出来了？', char, second_char)
                                char = {second_char:{'score':pred[i][second_char_index]}}
                                max_score = pred[i][second_char_index]
                        elif max_score <0.8 and ('▵'  not in char) and ('▿' not in char):  #如果分数<0。8 则记录需要做bigram的符号标记    将角标去掉
                            char = [{char:{'score':max_score},second_char:{'score',pred[i][second_char_index]}}]   '''
                # if type([2]) == type(char):    #一步一个坑
                #     word_list+=char
                # elif type({}) == type(char):
                #     word_list.append(char)
                # else:
                #     word_list.append({char:{'score':max_score}})

    if len(text_tmp_list)>1:
        text_tmp_1 = eng_error_correction(text_tmp_list, score_list_tmp, wrong_charindex_list,text_tmp)
        word_list+= text_tmp_1
    else:
        word_list += text_tmp_list
    aa = time.time()
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
                    # else:
                    #     print('j')


            word_bigram_score_list.append(word_bigram_score*p_pred**gamma) if len(path) > 2 else word_bigram_score_list.append(p_pred)   #
            score_list_final.append(word_bigram_score_path)
        max_score_index = np.argmax(np.array(word_bigram_score_list),axis=0)
        #print(''.join(paths[max_score_index]))
        final_score = score_list_final[max_score_index]
        final_text = ''.join(paths[max_score_index]).replace('卍','')
        #assert len(final_text)-final_text.count('▵') - final_text.count('▿') == len(final_score)
        final_text,final_score = strQ2B(final_text,final_score)
        #print('many path',time.time()-aa)
        return final_text,final_score
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
        final_text,final_score = strQ2B(final_text,score_list_final)
        strQ2B_text = final_text.replace('▿', ' ')

        strQ2B_text = strQ2B_text.replace('▵', '　')
        #print('one path',time.time()-aa)
        return strQ2B_text,final_score   #,score_list     ###score  等下再拿出来


def decode_ori_ori(pred):
    char_list = []
    score_list = []
    pred_text = pred.argmax(axis=1)
    #pred_text = np.argsort(pred,axis=1)
    #print(pred_text)
    max_score_index_list = []
    t2 = time.time()
    for i in range(len(pred_text)):
        #if pred_text[i] != nclass - 1: #and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1]))):
            max_score = pred[i][pred_text[i]]
         
            pred[i] = 0

            char_list.append(char_set[pred_text[i]])
            max_score_index_list.append(np.argmax(pred[i],axis=0))
            score_list.append(max_score)
    text = u''.join(char_list)
    strQ2B_text,score_list = strQ2B(text, score_list)
    strQ2B_text =  strQ2B_text.replace('▿', ' ')
    strQ2B_text = strQ2B_text.replace('▵', '　')
    return strQ2B_text,[str(ele) for ele in score_list]
def predict_batch(img,image_info):
    pre_time_a = time.time()
    y_pred = basemodel.predict_on_batch(img)[:,2:,:]
    pre_time = time.time() - pre_time_a
    Time0 = 0    
    Time1 = 0
    result_info = []
    #logging.info('chn batch')
    for i in range(len(y_pred)):
        #try:
        if True:
            a = time.time()
            text,scores = decode_chn_eng(y_pred[i])
            b = time.time()-a
            Time0 += b
            #print('加后处理时间',b)


        #except Exception as e:
            #logging.info(e)
            a = time.time()
            text1,scores1 = decode_ori_ori(y_pred[i])
            b = time.time()-a
            Time1 += b
            #print('不加后处理时间',b)
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
