import re
import json
import pickle
import logging
import itertools
import numpy as np



def file_len(path):
    """
    :param file_path:
    :return len(file):
    """
    with open(path, "r") as file:
        return sum([1 for i in file])


class decode_ctc():

    def __init__(self, k=1e-8, **kwargs):

        with open(kwargs.get('eng_dict_path_file'), 'rb') as pkl_file, open(
                kwargs.get("lfreq_chn_word_path")) as lfreq_chn_word_file, \
                open(kwargs.get("lfreq_jap_word_path")) as lfreq_jap_word_file:
            self.word_dict = pickle.load(pkl_file)
            self.word_dict['fl'] = 1
            self.word_dict['pdw'] = 1
            self.word_dict['cre'] = 1
            self.lfreq_chn_word = json.loads(lfreq_chn_word_file.read())  #
            logging.info(' chn word file lodding done')
            self.lfreq_jap_word = json.loads(lfreq_jap_word_file.read())
            logging.info(' jap word file lodding done')
        self.k = k

        self.gamma = 2 # 发射概率的阈值
        self.alpha = 1  # LM 的阈值
        self.easy_confused_gamma = 1
        self.easy_confused_alpha = 1

        self.Easily_confused_word = {'径':{'真径':'直径'}}
        self.Easily_confused_hard = ['人','入']
        self.Easily_confused = []

        #词库替换
        self.look_up_table = {
                                'Lanvatinib':'Lenvatinib',
                                'ジャスビア錠':'ジャヌビア錠',
                                'ピップロロールフマ':'ビソプロロールフマ',
                                '果急入院':'緊急入院',
                                '尿識':'尿量',
                                'レンバテニブ':'レンバチニブ',
                                '急性管腸炎':'急性胃腸炎'
                                }

        self.wrong_char_num = 5

        self.characters_num_per_paper = 0  # 统计一整页文字有多少个
        self.wrong_characters_num_per_paper = 0  # 统计一整页文字有多少个是概率不确定的


    @staticmethod
    def init_char_set(file_path, ):
        file_path_len = file_len(file_path)
        with open(file_path, "r") as file:
            # for i,line in tqdm(enumerate(file),total=file_path_len):
            char_set = [c.strip('\n') for c in file]
        char_set.append('卍')
        nclass = len(char_set)
        return char_set, nclass

    @staticmethod
    def isalpha(c):
        if c <= 'z' and c >= 'a' or c >= 'A' and c <= 'Z':
            return True
        return False

    @classmethod
    def filter_blank(cls, word, score_list):
        if len(word) - word.count('▿') - word.count('▵') != len(score_list):
            #print('一开始就不相等')
            #print(word, score_list)
            return word, score_list
        else:
            word_list = list(word)
            # word_tmp_list = word.split(' ')
            # space_num = 0
            for index, w in enumerate(word):
                if w == ' ':
                    if index == 0 or index == len(word) - 1:  # 判断句首和句尾
                        word_list[index] = ''
                        score_list[index] = ''
                    else:
                        if cls.is_chinese(word[index - 1]) or cls.is_chinese(word[index + 1]):  # 如果前后出现汉字 空格去掉
                            word_list[index] = ''
                            score_list[index] = ''
                        elif word[index + 1] == ' ':  # 如果后面是空格，则 空格去掉
                            word_list[index + 1] = ''
                            score_list[index + 1] = ''
            text = ''.join(list(filter(None, word_list)))
            if len(text) == 1 and text[0] == ' ':
                return '', []
            return text, list(filter(None, score_list))

    @staticmethod
    def is_chinese(word):
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False
    def is_jap(self,word):
        jap_num = 0
        for ch in word:
            # if '\u0800' <= ch <= '\u4e00':
            #    jap_num += 1
            if '\u3040' <= ch <= '\u309F':  # Hiragana
                return True
            if '\u30A0' <= ch <= '\u30FF':  # Katakana
                return True
            # 与汉字重叠了，只看片假名吧
            if '\u4E00' <= ch <= '\u9FBF': #Kanji
            # jap_num += 1
                return True
        return False

    @staticmethod
    def get_list_score(score_list):
        score = 1
        for i in score_list:
            score *= i
        return score

    def eng_error_correction(self, text_tmp_list, score_list_tmp, wrong_charindex_list, text_tmp):
        tmp_word_list = []
        tmp_dict = {}
        if len(wrong_charindex_list) == 0:
            score = self.get_list_score(score_list_tmp)
            return [{text_tmp: {'score': score, 'score_list': score_list_tmp}}]
        for i in itertools.product(*text_tmp_list):
            word = ''.join(list(i))
            # if '卍' in word:  # 如果占位符在还得减去一个score 麻烦
            score = 1
            if word.replace('卍', '').lower() in self.word_dict:  # 将占位符过滤掉
                score_list = []
                for index, w in enumerate(word):
                    score *= text_tmp_list[index][w]['score']
                    if w != '卍':  # 将占位符的概率也加进去计算
                        score_list.append(text_tmp_list[index][w]['score'])
                tmp_dict[word.replace('卍', '')] = {'score': score, 'score_list': score_list}
                #print('word 转换前', text_tmp)
                #print('word 转换后', word.replace('卍', ''))
        if tmp_dict == {}:
            tmp_dict[text_tmp] = {'score': decode_ctc.get_list_score(score_list_tmp), 'score_list': score_list_tmp}
        tmp_word_list.append(tmp_dict)
        if len(tmp_word_list) > 0:
            # print('转换前',text_tmp)
            # print('转换后',tmp_word_list)
            return tmp_word_list
        else:
            score = 1
            for s in score_list_tmp:
                score *= s
            return [{text_tmp: {'score': score, 'score_list': score_list_tmp}}]

    def get_word_bigram_score(self, word_list,lan):  # 没考虑开头和结尾

        if lan.upper() =="JAP" or lan.upper() == 'JPE':
            lfreq_word = self.lfreq_jap_word
        else:
            lfreq_word = self.lfreq_chn_word
        score = 1
        for i, word in enumerate(word_list):
            if word != ' ':
                word_combine = ''
                if i == 0:
                    word_combine = word
                    score = 1 * 1
                else:
                    for j in range(1, i + 1):
                        if word_list[i - j] != ' ' and word_list[i - j] != '卍':
                            word_combine = word_list[i - j] + ' ' + word  # 这一步是要得到当前词与前面不是空格的词的组合
                            break
                # print('word_combine',word_combine)
                if word_combine in lfreq_word and i != 0 and word in lfreq_word:
                    #print('word_combine{}  freq {} score {}'.format(word_combine,lfreq_word[word_combine],lfreq_word[word_combine] * 1.0 /lfreq_word[word]))
                    score = score * lfreq_word[word_combine] * 1.0 /lfreq_word[word]
                else:
                    score = score * self.k
        return score

    @classmethod
    def strQ2B(cls, a, max_score_list):
        # t4 = time.time()
        is_chinese_or = cls.is_chinese(a)
        # print('判断是否是中文时间：',time.time() - t4)
        if is_chinese_or:
            a = a.replace('(', '（')
            a = a.replace(')', '）')
            a = a.replace(',', '，')
            a = a.replace(':', '：')
            a = a.replace('?', '？')
            a = a.replace(';','；')
            list_a = list(a)
            for i in re.finditer('\.|\(|（|）|\)', a):
                if list_a[i.start()] == '.':
                    if max_score_list[i.start()] < 0.9:
                        if cls.is_chinese(a[i.start() - 1]):  # 判定中文
                            list_a[i.start()] = '。'
                        elif a[i.start() - 1] == ')' and (
                                (ord(a[i.start() - 2]) > 47 and ord(a[i.start() - 2]) < 59) or cls.is_chinese(
                            a[i.start() - 2])):
                            list_a[i.start()] = '。'
                        elif a[i.start() - 1] == '）':
                            list_a[i.start()] = '。'
                        elif a[i.start() - 1] == '）':
                            list_a[i.start()] = '。'
            a = ''.join(list_a)
        else:
            a = a.replace('（', '(')
            a = a.replace('）', ')')
            a = a.replace('：', ':')
            a = a.replace('？', '?')
            a = a.replace('；', ';')
            a = a.replace('，',',')
            # a = a.replace('。','.')
        #a = cls.filter_blank(a, max_score_list)
        return a,max_score_list

    def decode_ori(self, pred,char_set,lan):
        nclass = len(char_set)
        char_list = []
        score_list = []
        pred_text = pred.argmax(axis=1)
        for i in range(len(pred_text)):
            if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1]))):
                max_score = pred[i][pred_text[i]]
                char_list.append(char_set[pred_text[i]])
                score_list.append(max_score)
        text = u''.join(char_list)
        if lan.upper() == 'CHN':
            text,score_list = decode_ctc.strQ2B(text, score_list)
        text = text.replace('▿', ' ')
        text = text.replace('▵', '　')
        erro_record = self.count_error_characters(score_list)
        return text, [str(ele) for ele in score_list],erro_record

    def char_in_Easily_confused_word(self,s1, s2):

        if s1 + s2 in self.Easily_confused_word[s2]:
            # if s1 + s2 in Easily_confused_word[s2]:
            # print('转换前', s1 + s2)
            # print('转换后', Easily_confused_word[s2][s1 + s2])
            return self.Easily_confused_word[s2][s1 + s2]

    def count_error_characters(self,final_score):
        return {'wrong_characters_num':len(np.where(np.array(final_score)<0.6)[0]),'characters_num':len(final_score)}

    def find_best_path(self,paths,lan,word_list):
        word_bigram_score_list = []
        score_list_final = []
        for path in paths:
            word_bigram_score_path = []
            word_bigram_score = self.get_word_bigram_score(path, lan) ** self.alpha
            path = list(path)
            p_pred = 1
            for j in range(len(path)):
                score_path = word_list[j][path[j]]['score']  # 获得每个词的分数
                p_pred *= score_path
                if 'score_list' in word_list[j][path[j]]:
                    word_bigram_score_path += word_list[j][path[j]]['score_list']
                else:
                    # if path[j] != '卍':
                    word_bigram_score_path += [word_list[j][path[j]]['score']]  # 获得每个path中每个字符分数列表
            word_bigram_score_list.append(word_bigram_score * p_pred ** self.gamma) if len(
                path) > 2 else word_bigram_score_list.append(p_pred)  #
            score_list_final.append(word_bigram_score_path)
        max_score_index = np.argmax(np.array(word_bigram_score_list), axis=0)
        final_score = score_list_final[max_score_index]
        final_text = ''.join(paths[max_score_index]).replace('卍', '')
        if lan.upper() == 'CHN':
            final_text, final_score = decode_ctc.strQ2B(final_text, final_score)
        final_text = final_text.replace('▿', ' ')
        final_text = final_text.replace('▵', '　')
        return final_text,final_score

    def get_one_path_text(self,paths,lan,word_list):
        score_list_final = []
        path = list(paths[0])
        for j in range(len(path)):
            if 'score_list' in word_list[j][path[j]]:
                score_list_final += word_list[j][path[j]]['score_list']
            else:
                score_list_final.append(word_list[j][path[j]]['score'])
        final_text = ''.join(list(paths[0]))
        final_score = score_list_final
        if lan.upper() == 'CHN':
            final_text, final_score = decode_ctc.strQ2B(final_text, score_list_final)
        strQ2B_text = final_text.replace('▿', ' ')
        strQ2B_text = strQ2B_text.replace('▵', '　')
        return strQ2B_text,final_score

    def decode_chn_eng(self,pred,lan,char_set):
        nclass = len(char_set)
        #print(pred.shape)
        pred_text = pred.argmax(axis=1)
        text_tmp = ''  # 存放临时单词
        word_score_tmp = 1
        text_tmp_list = []  # 存放临时的字符 + score
        score_list_tmp = []  # 存放tmp Max score
        wrong_charindex_list = []  # 存放tmp 嫌疑字符列表
        wrong_charindex = 0
        word_list = []  # 存放word
        wrong_word_index_list = []
        word_wrong_num_sianal = True
        for i in range(len(pred_text)):
            if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1]))):
                max_score = pred[i][pred_text[i]]
                pred[i][pred_text[i]] = 0
                char = char_set[pred_text[i]]
                self.characters_num_per_paper += 1    # 统计一整页文字有多少个
                if decode_ctc.isalpha(char):  #如果是字母
                    if max_score < 0.9 and  len(wrong_charindex_list) < self.wrong_char_num : # 限制错误字符的个数 如果超过三个 这一步也不计算第二个字符了
                        wrong_charindex_list.append(wrong_charindex)  # 嫌疑字index列表

                        second_char_index = pred[i].argmax(axis=0)

                        text_tmp_list.append({char: {'score': max_score},
                                              char_set[second_char_index]: {'score': pred[i][second_char_index]}})
                    else:
                        text_tmp_list.append({char: {'score': max_score}})
                    text_tmp += char
                    word_score_tmp *= max_score
                    score_list_tmp.append(max_score)
                    wrong_charindex += 1
                else:
                    if max_score < 0.6 :  # 优化字符识别成特殊字符的情况
                        second_char_index = pred[i].argmax(axis=0)
                        second_char = char_set[second_char_index]
                        if decode_ctc.isalpha(second_char) and len(wrong_charindex_list) <self.wrong_char_num and ('▵' not in char) and (
                                '▿' not in char):#  如果第二个字符是字母
                            wrong_charindex_list.append(wrong_charindex+1)
                            text_tmp += char
                            score_list_tmp.append(max_score)
                            if second_char != '卍' and second_char != ' ':
                                text_tmp_list.append(
                                    {char: {'score': max_score}, second_char: {'score': pred[i][second_char_index]}})
                            else:
                                text_tmp_list.append(
                                    {char: {'score': max_score}})

                            continue
                    if len(text_tmp_list) > 1: #存放临时的字符 + score  这是对存放的单词的处理
                        if text_tmp.lower() in self.word_dict:  # 如果是单词 继续
                            score_list_tmp = score_list_tmp
                            text_tmp_1 = [
                                {text_tmp: {'score': word_score_tmp, 'score_list': score_list_tmp}}]
                            need_bigram = False
                        else:  # 如果不是单词 纠错
                            if len(wrong_charindex_list) < self.wrong_char_num+1: #如果错误字符小于3 纠错    英文单词就错
                                text_tmp_1 = self.eng_error_correction(text_tmp_list, score_list_tmp, wrong_charindex_list,
                                                                  text_tmp)  # 得到修正后的单词列表
                                if len(text_tmp_1) > 1:
                                    wrong_word_index_list.append(0)
                            else:   #如果错误字符数超过限制 则不纠错
                                text_tmp_1 = [{text_tmp: {'score': word_score_tmp, 'score_list': score_list_tmp}}]
                            # eng_error_time_b = time.time()
                        word_list += (text_tmp_1)
                        text_tmp = ''
                        word_score_tmp = 1
                        wrong_charindex = 0
                        wrong_charindex_list = []
                        score_list_tmp = []
                        text_tmp_list = []
                    elif len(text_tmp_list) == 1:  #如果是一个字符 有可能特殊符号 数字 方块字这样的
                        if '卍' in text_tmp_list[0]:
                            del text_tmp_list[0]['卍']  #如果是占位符 便不做纠错
                        word_list += text_tmp_list

                        text_tmp = ''
                        wrong_charindex = 0
                        wrong_charindex_list = []
                        score_list_tmp = []
                        text_tmp_list = []
                    # if is_chinese(char): #对汉字的处理
                    if char in self.Easily_confused_word:  # 如果在易混淆词库
                        if word_list!=[]:
                            s1 = list(word_list[-1].keys())[0]
                            if len(word_list) > 0 and self.char_in_Easily_confused_word(s1, char):
                                tmp_s = self.char_in_Easily_confused_word(s1, char)
                                word_list[-1][tmp_s[0]] = word_list[-1][s1]
                                word_list[-1].pop(s1)  # 将原字符删除掉
                        word_list.append({char: {'score': max_score}})
                    elif ((char in self.Easily_confused_hard and max_score < 0.95) or (max_score < 0.6 and ('▵' not in char) and ('▿' not in char))) \
                            and len(wrong_word_index_list) < self.wrong_char_num and self.is_jap(char):  # 如果字符是易混淆字符且概率小于0。95 或者最大值小于0。6
                        wrong_word_index_list.append(0)
                        second_char_index = pred[i].argmax(axis=0)  # 这里备用字符不做是否占位符的判断
                        if second_char_index != nclass - 1 and second_char_index != 0:
                            second_char = char_set[second_char_index]
                            char = {char: {'score': max_score}, second_char: {'score': pred[i][second_char_index]}}
                        else:
                            char = {char: {'score': max_score}}
                        word_list.append(char)
                    # elif( char in self.Easily_confused and max_score < 0.6 and ('▵' not in char)
                          # and ('▿' not in char)) and len(wrong_word_index_list) < self.wrong_char_num:

                    else:
                        word_list.append({char: {'score': max_score}})
        if len(text_tmp_list) > 1 and len(wrong_charindex_list) < self.wrong_char_num+1 and len(wrong_word_index_list)<self.wrong_char_num :  #如果最后一个单词错误的字符数小于3
            text_tmp_1 = self.eng_error_correction(text_tmp_list, score_list_tmp, wrong_charindex_list, text_tmp)
            if len(text_tmp_1) > 1 :
                wrong_word_index_list.append(0)
        else:
            text_tmp_1 = [
                {text_tmp: {'score': word_score_tmp, 'score_list': score_list_tmp}}]
        word_list += text_tmp_1
        word_list = list(filter(None, word_list))  # 必须要过滤 否则path为空
        paths = list(itertools.product(*word_list))
        if len(paths) > 1:
            final_text,final_score = self.find_best_path(paths, lan, word_list)
            erro_record = self.count_error_characters(final_score)
            return final_text, final_score,erro_record
        elif len(paths) == 1:
            final_text,final_score = self.get_one_path_text( paths, lan, word_list)
            erro_record = self.count_error_characters(final_score)

        ### 词库替换
        final_text = self.replace_look_up_table(final_text)

        return final_text, final_score,erro_record  # ,score_list     ###score  等下再拿出来

    def replace_look_up_table(self,text):
        '''根据替换字典，替换
        '''
        for k, v in self.look_up_table.items():
            newtext = text.replace(k, v)
            if(newtext!=text): #一旦命中字典则不再往下遍历
                text = newtext
                break
            text = newtext
        return text 

if __name__ == '__main__':
    import os, time

    kwargs = {}
    #rrrr =
    #'/fengjing/data_script/OCR_textrender/data/chars/eng_new.txt'  chn7213.txt

    DCTC = decode_ctc(eng_dict_path_file='/fengjing/dip_server/ocr_data_labeling/eng_dict.pkl',
                            # lfreq_chn_word_path='./char_rec/corpus/char_and_word_bigram_chneng.json',
                            # lfreq_jap_word_path='./char_rec/corpus/char_and_word_bigram_jap.json')
                            lfreq_chn_word_path='/fengjing/dip_server/ocr_data_labeling/count_word_chn0.json',
                            lfreq_jap_word_path='/fengjing/dip_server/ocr_data_labeling/count_word_chn0.json')

    char_set = open('/fengjing/data_script/OCR_textrender/data/chars/japeng.txt', 'r', encoding='utf-8').readlines()
    # char_set = open('/fengjing/data_script/OCR_textrender/data/chars/eng_new.txt', 'r', encoding='utf-8').readlines()
    char_set = [c.strip('\n') for c in char_set] +['卍']
    #char_set.append('卍')
    npyPath = '/fengjing/test_img/npy/'
    npyList = os.listdir(npyPath)
    Time0 = 0
    Time1 = 0
    num = 0
    for npy in npyList:
        # if 'npy' in npy:
        if npy.endswith('npy'):
            num += 1
            # print(num)
            preds = np.load(os.path.join(npyPath, npy))
            # preds = np.load(os.path.join(npyPath, '1582801674.5322697.npy'))
            # preds = np.load('/fengjing/test_img/1.npy')1582785234.6443508.npy'
            # print(preds.shape)
            for i in range(len(preds)):
                pred = preds[i]
                #print(pred.shape)
                pred_ctc = pred.copy()
                a = time.time()
                text, score,erro_record = DCTC.decode_chn_eng(pred,'jap',char_set)
                b = time.time()
                Time0 = Time0 + b - a
                #print('加后处理时间:', b - a)
                if len(text) - text.count('▵') - text.count('▿') != len(score):
                    print('不一样啊啊啊啊')
                # print(text[0])
                # if len(text) -text.count('　') - text.count(' ') != len(scores):
                #     print('又不一样了',text,scores)
                #     print(len(text) -text.count('　') - text.count(' '),len(scores))

                # print('decode viterbi',)
                # b= time.time()
                # print('decode time',b-a)
                # #print('decode Viterbi',decode_Viterbi(pred1[0]))
                #
                c = time.time()
                text_ctc,score,erro_record = DCTC.decode_ori(pred_ctc,char_set,'chn')
                # print('不加后处理时间:',time.time()-c)
                # print(text_ctc)
                if text_ctc != text:
                    print('转换前', text_ctc)
                    print('转换后', text)
                    print(npy)
                    print(i)
                # print('ctc_decode',)
                d = time.time()
                Time1 = Time1 + d - c
                #print('ctc decode time', d - c)
            #     break
            # break
    print('加后处理 总时间', Time0)
    print('不加后处理总时间', Time1)

