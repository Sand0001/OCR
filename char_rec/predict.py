import os
import json
import time
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps,ImageDraw,ImageFont
from char_rec.decode import decode_ctc
from char_rec import shufflenet_res_crnn as densenet
import traceback
#from char_rec import dl_resnet_crnn_cudnnlstm as resnet

decode_ctc = decode_ctc(eng_dict_path_file='./char_rec/corpus/eng_dict.pkl',
                      lfreq_chn_word_path='./char_rec/corpus/char_and_word_bigram_chneng.json',
                      lfreq_jap_word_path='./char_rec/corpus/char_and_word_bigram_jap.json')
                      #lfreq_chn_word_path='./char_rec/corpus/count_word_chn0.json',
                      #lfreq_jap_word_path='./char_rec/corpus/count_word_chn0.json')
graph = tf.get_default_graph()
class predict():
    def __init__(self,**kwargs):

        self.chn_charset = predict.get_charset(kwargs.get('chn_charset_path'))
        self.eng_charset = predict.get_charset(kwargs.get('eng_charset_path'))
        self.jap_charset = predict.get_charset(kwargs.get('jap_charset_path'))

        self.eng_model = self.load_model(kwargs.get('eng_model_path'),'eng')
        self.chn_model = self.load_model(kwargs.get('chn_model_path'),'chn')
        self.jap_model = self.load_model(kwargs.get('jap_model_path'),'jap')


        # self.chn_font = ImageFont.truetype('/data/fengjing/ocr_recognition_test/fonts/chn/华文宋体.ttf',36)
        # self.eng_font = ImageFont.truetype('/data/fengjing/ocr_recognition_test/fonts/eng/Times New Roman.ttf',36)
        # self.jap_font = ImageFont.truetype('/data/fengjing/ocr_recognition_test/fonts/jap/ToppanBunkyuGothicPr6N.ttc', 36)

        self.predict_time = 0
        self.decode_time = 0

        self.color_normal = (0, 0, 0)
        self.color_red = (255, 0, 0)
        self.rec_pic_path = '/data/fengjing/ocr_recognition_test/html/image_rec/'



    @staticmethod
    def get_charset(charset_path,):
        char_set = open(charset_path, 'r', encoding='utf-8').readlines()
        char_set = [c.strip('\n') for c in char_set]
        char_set.append('卍')
        return char_set


    def load_model(self,model_path,lan):
        if lan.upper() == 'JAP' or lan.upper() =='JPE':
            nclass = len(self.jap_charset)

        elif lan.upper() == 'CHN':
            nclass = len(self.chn_charset)
        elif lan.upper() == 'ENG':
            nclass = len(self.eng_charset)
        else:
            nclass  = len(self.chn_charset)

        mult_model, basemodel = densenet.get_model(False, 32, nclass)
        modelPath = os.path.join(os.getcwd(),
                                 model_path)  # weights_eng_finetune_300_finally_resnet-01-1.11.h5
        if os.path.exists(modelPath):
            basemodel.load_weights(modelPath)
            print('{} shufflenet model loading done'.format(lan))
        else:
            print('NO {} model exist'.format(lan))
        return basemodel

    def gen_rec_img(self,scores,text,lan,picname):
        if lan.upper() == 'JAP' or lan.upper() =='JPE' :
            font = ImageFont.truetype('/data/fengjing/ocr_recognition_test/fonts/jap/ToppanBunkyuGothicPr6N.ttc', 36)


        elif lan.upper() == 'CHN' :
            font = ImageFont.truetype('/data/fengjing/ocr_recognition_test/fonts/chn/华文宋体.ttf',36)

        elif lan.upper() == 'ENG' :
            font = ImageFont.truetype('/data/fengjing/ocr_recognition_test/fonts/eng/Times New Roman.ttf',36)
        else:
            font = ImageFont.truetype('/data/fengjing/ocr_recognition_test/fonts/chn/华文宋体.ttf',36)

        width = font.getsize(text)[0]
        im = Image.new("RGB", (width + 20, 46), (255, 255, 255))
        draw = ImageDraw.Draw(im)

        if len(scores)>0:
            if len(np.where(np.array(scores)<0.9)[0]) == 0:
               draw.text((5,5), text, fill=self.color_normal, font=font)
            else:

                    start_x = 5
                    for index,t in enumerate(text):
                        if scores[index]<0.6:

                            draw.text((start_x,5), t, fill=self.color_red, font=font)
                        else:
                            draw.text((start_x,5), t, fill=self.color_normal, font=font)
                        start_x += font.getsize(t)[0]

        im.save(os.path.join(self.rec_pic_path,'rec_'+ picname))

    def get_basemodel(self,lan):
        if lan.upper() == 'JAP' or lan.upper() == 'JPE':
            basemodel = self.jap_model
            char_set = self.jap_charset

        elif lan.upper() == 'CHN':
            basemodel = self.chn_model
            char_set = self.chn_charset
        elif lan.upper() == 'ENG':
            basemodel = self.eng_model
            char_set = self.eng_charset
        else:
            basemodel = self.chn_model
            char_set = self.chn_charset
        return basemodel,char_set

    def predict_batch(self, img, image_info, lan):

        basemodel,char_set  = self.get_basemodel(lan)
        a = time.time()
        global graph
        with graph.as_default():
            y_pred = basemodel.predict_on_batch(img)[:, 2:, :]
        self.predict_time += time.time() - a
        result_info = []
        # logging.info('chn batch')
        for i in range(len(y_pred)):
            b = time.time()
            try:
                text, scores = decode_ctc.decode_chn_eng(y_pred[i], lan, char_set)
            except:
                text, scores = decode_ctc.decode_ori(y_pred[i],lan,char_set)
            self.decode_time += time.time() - b
            imagename = {}
            imagename['location'] = image_info[i]['location']
            imagename['text'] = text
            imagename['scores'] = [str(ele) for ele in scores]
            result_info.append(imagename)
        return result_info

    def get_json_path(self,lan):   # fengjing test 服务用的
        if lan == 'JPE':
            json_label_path = '/data/fengjing/ocr_recognition_test/label_json_jap/'
        elif lan.upper() == 'CHN':
            json_label_path = '/data/fengjing/ocr_recognition_test/label_json_chn/'
        elif lan.upper() == 'ENG':
            json_label_path = '/data/fengjing/ocr_recognition_test/label_json_eng/'
        else:
            json_label_path = '/data/fengjing/ocr_recognition_test/label_json_chn/'
        return json_label_path

    def predict_batch_test(self,img, image_info,lan):
        erro_record_batch = {"wrong": 0, 'all': 0}
        basemodel, char_set = self.get_basemodel(lan)

        a = time.time()
        global graph
        with graph.as_default():
            y_pred = basemodel.predict_on_batch(img)[:, 2:, :]
        self.predict_time += time.time() - a
        result_info = []
        # logging.info('chn batch')
        for i in range(len(y_pred)):
            b = time.time()
            y_pred_1 = y_pred[i].copy()
            try:
               text, scores,erro_record = decode_ctc.decode_chn_eng(y_pred[i],lan,char_set)
            except:
               text, scores,erro_record = decode_ctc.decode_ori(y_pred[i],char_set,lan)

            self.decode_time += time.time() - b
            text_ori, scores_ori,erro_record = decode_ctc.decode_ori(y_pred_1, char_set, lan)
            erro_record_batch['wrong'] += erro_record['wrong_characters_num']
            erro_record_batch['all'] += erro_record['characters_num']
            text = text.replace('　','▵')
            text = text.replace(' ','▿')

            text_ori = text_ori.replace('　', '▵')
            text_ori = text_ori.replace(' ', '▿')

            imagename = {}
            label_and_rec_text = {}
            label_and_rec_text['label'] = ''
            label_and_rec_text['rec_text'] = text
            imagename['text'] = label_and_rec_text
            imagename['location'] = image_info[i]['location']
            imagename['rec_img'] = 'rec_' + image_info[i]['picname']
            imagename['img_name'] = image_info[i]['picname']
            #imagename['text'] = text
            imagename['scores'] = [str(ele) for ele in scores]
            little_pic_json_file = image_info[i]['picname'][:-4]+'.json'
            label_data = {'imgurl':'http://39.104.88.168/image_rec/'+ image_info[i]['picname'],'text':text,'is_save':'true'}
            json_label_path = self.get_json_path(lan)
            # with open(json_label_path+little_pic_json_file,'w') as f:
            #     json.dump(label_data,f)
            if text_ori != text:
                label_and_rec_text['label'] = text_ori
            try:
                self.gen_rec_img(scores,text,lan,image_info[i]['picname'])
            except:
                continue
            result_info.append(imagename)
        print('erro_record_batch',erro_record_batch)
        return result_info, erro_record_batch

    def predict_batch_v2(self,img, image_info,lan):

        basemodel, char_set = self.get_basemodel(lan)
        a = time.time()
        global graph
        with graph.as_default():
            y_pred = basemodel.predict_on_batch(img)[:, 2:, :]
        self.predict_time += time.time() - a
        result_info = []
        erro_record_batch = {"wrong":0,'all':0}
        # logging.info('chn batch')
        for i in range(len(y_pred)):
            width = 0

            for jj,slice_img in enumerate(image_info[i]):
                b = time.time()
                slice_width = width+slice_img['image'].shape[1]+2
                try:
                   text, scores,erro_record = decode_ctc.decode_chn_eng(y_pred[i][width//4:slice_width//4],lan,char_set)
                except:
                    logging.info('error:{}'.format(traceback.format_exc()))
                    text, scores,erro_record = decode_ctc.decode_ori(y_pred[i][width//4:slice_width//4],lan,char_set)
                self.decode_time += time.time() - b
                width += slice_width
                imagename = {}
                imagename['location'] = slice_img['location']
                imagename['text'] = text
                imagename['scores'] = [str(ele) for ele in scores]
                result_info.append(imagename)
                erro_record_batch['wrong'] += erro_record['wrong_characters_num']
                erro_record_batch['all'] += erro_record['characters_num']
        return result_info,erro_record_batch
if __name__ == '__main__':
    predict = predict()
