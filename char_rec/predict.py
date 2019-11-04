import os
import tensorflow as tf
from char_rec.decode import decode_ctc
from char_rec import shufflenet_res_crnn as densenet


decode_ctc = decode_ctc(eng_dict_path_file='./char_rec/corpus/eng_dict.pkl',
                      #lfreq_chn_word_path='./char_rec/corpus/count_word_chn0.json',
                      #lfreq_jap_word_path='./char_rec/corpus/count_word_chn0.json')
                      lfreq_chn_word_path='./char_rec/corpus/char_and_word_bigram_chneng.json',
                      lfreq_jap_word_path='./char_rec/corpus/char_and_word_bigram_jap.json')
graph = tf.get_default_graph()
class predict():
    def __init__(self,**kwargs):

        self.chn_charset = predict.get_charset(kwargs.get('chn_charset_path'))
        self.eng_charset = predict.get_charset(kwargs.get('eng_charset_path'))
        self.jap_charset = predict.get_charset(kwargs.get('jap_charset_path'))

        self.eng_model = self.load_model(kwargs.get('eng_model_path'),'eng')
        self.chn_model = self.load_model(kwargs.get('chn_model_path'),'chn')
        self.jap_model = self.load_model(kwargs.get('jap_model_path'),'jap')



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
        else:
            print('NO {} model exist'.format(lan))
        return basemodel

    def predict_batch(self,img, image_info,lan):
        if lan.upper() == 'JAP' or lan.upper() =='JPE':
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
        global graph
        with graph.as_default():
            y_pred = basemodel.predict_on_batch(img)[:, 2:, :]

        result_info = []
        # logging.info('chn batch')
        for i in range(len(y_pred)):
            #try:
            #    text, scores = decode_ctc.decode_chn_eng(y_pred[i],lan,char_set)
            #except:
            #    text, scores = decode_ctc.decode_ori(y_pred[i])
            text, scores = decode_ctc.decode_ori(y_pred[i],char_set,lan)
            # if text != text_ori:

            imagename = {}
            imagename['location'] = image_info[i]['location']
            imagename['text'] = text
            imagename['scores'] = [str(ele) for ele in scores]
            result_info.append(imagename)
        return result_info
if __name__ == '__main__':
    predict = predict()
