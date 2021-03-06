import os
import json
import threading
import numpy as np
from PIL import Image
import traceback
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers import CuDNNLSTM
from keras.layers.merge import add, concatenate
from keras.optimizers import Adadelta
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.layers.wrappers import TimeDistributed
from . import resnet
#from parameter import *
#K.set_learning_phase(0)


# GPU_ID_LIST = '0,1,3,4,5,6,7'
# os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID_LIST

img_h = 32
img_w = 280
batch_size = 128
maxlabellength = 25
# GPU_NUM = len(GPU_ID_LIST.split(','))
GPU_NUM = 1
batch_size = 112 * GPU_NUM
#batch_size = 2
#train_size = 500000
#test_size = 40000
#tag = 'multilan_test_v11'
#train_size = 6300000
train_size = 6000000
test_size = 40000
tag = 'chn_eng_eroded_final_v2'

encode_dct =  {}


def get_session(gpu_fraction=0.1):

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False))

def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        try:
            first_whitespace_idx = i.index(' ')
        except:
            continue
        img_name = i[0 :  first_whitespace_idx].strip(':').zfill(8) +  '.jpg'
        if len(i[first_whitespace_idx + 1:]) == 0 or len(i[first_whitespace_idx + 1:]) > maxlabellength or len(img_name) == 0 :
            continue
        #p = i.split(' ')
        dic[img_name] = i[first_whitespace_idx + 1 :]
    return dic

class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """
    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0
    def get(self, batchsize):
        r_n=[]
        if(self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index : self.index + batchsize]
            self.index = self.index + batchsize

        return r_n

cur_line = None



def gen(data_file, image_path, batchsize=128, maxlabellength=32, imagesize=(32, 280)):
    image_label = readfile(data_file)
    _imagefile = [i for i, j in image_label.items()]
    #x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    x = np.zeros((batchsize, imagesize[1], imagesize[0], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * 10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)
    #idx = 0
    while 1:
            shufimagefile = _imagefile[r_n.get(batchsize)]
            for idx, fname in enumerate(shufimagefile):
            #for i in range(0, len(r_n.range)):
                #fname = _imagefile[i]
                img_f = os.path.join(image_path, fname).strip(':')
                img1 = Image.open(img_f).convert('L')
                #width, height = img1.size[0], img1.size[1]
                #scale = height * 1.0 / img_h
                #width = int(width / scale)
                #resized_img = img1.resize((img_w, img_h), Image.ANTIALIAS)
                img = np.array(img1, 'f') / 255.0 - 0.5
                #转成w * h
                x[idx] = np.expand_dims(img, axis=2).swapaxes(0,1)
                label = image_label[fname]
                label_idx_list = [encode_dct[c] for c in label]
                label_length[idx] = len(label_idx_list)
                #不太明白这里为什么要减去2
                #跟两个MaxPooling有关系?
                input_length[idx] = imagesize[1] // 4 - 2
                #labels[idx, :len(str)] = [int(k) - 1 for k in str]
                labels[idx, :len(label_idx_list)] = label_idx_list
            inputs = {'the_input': x,
                        'the_labels': labels,
                        'input_length': input_length,
                        'label_length': label_length,
                        }
            outputs = {'ctc': np.zeros([batchsize])}
                    #print (new_input_length, new_label_length, new_labels.shape, new_labels)
            yield (inputs, outputs)

# # Loss and train functions, network architecture
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    a = 0.25
    r = 0.5
    y_pred = y_pred[:, 2:, :]
    loss =  K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    p = K.exp(-1 * loss)
    focalLoss = a * K.pow(1 - p, r) * loss
    return focalLoss



def get_model(training, img_h, nclass):
    input_shape = (None, img_h, 1)     # (128, 64, 1)
    # Make Networkw
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)
    inner = resnet.ResNet50(include_top=False, weights = None, input_tensor = inputs)
    # Convolution layer (VGG)
        # CNN to RNN
    #inner = Reshape(target_shape=((32, 2048)), name='reshape')(inner)  # (None, 32, 2048)
    inner = TimeDistributed(Flatten(), name='flatten')(inner)
    #inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)
    
    lstm_unit_num = 256

    # RNN layer
    lstm_1 = CuDNNLSTM(lstm_unit_num, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)  # (None, 32, 512)
    lstm_1b = CuDNNLSTM(lstm_unit_num, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
    lstm1_merged = add([lstm_1, lstm_1b])  # (None, 32, 512)
    lstm1_merged = BatchNormalization()(lstm1_merged)
    
    #lstm1_merged = Dropout(0.1)(lstm1_merged)

    lstm_2 = CuDNNLSTM(lstm_unit_num, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    lstm_2b = CuDNNLSTM(lstm_unit_num, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
    lstm2_merged = concatenate([lstm_2, lstm_2b])  # (None, 32, 1024)
    lstm_merged = BatchNormalization()(lstm2_merged)

    #lstm_merged = Dropout(0.1)(lstm_merged)


    # transforms RNN output to character activations:
    inner = Dense(nclass, kernel_initializer='he_normal',name='dense2')(lstm2_merged) #(None, 32, 63)
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[None], dtype='float32') # (None ,8)
    input_length = Input(name='input_length', shape=[1], dtype='int64')     # (None, 1)
    label_length = Input(name='label_length', shape=[1], dtype='int64')     # (None, 1)

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) #(None, 1)
    model = None
    if training:
        model =  Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    else:
        model = Model(inputs=inputs, outputs=y_pred)
        model._make_predict_function()
        return model, model
    model.summary()
    multi_model = multi_gpu_model(model, gpus=GPU_NUM)
    save_model = model
    ada = Adadelta()
    #multi_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])
    multi_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada, metrics=['accuracy'])
    return save_model, multi_model


if __name__ == '__main__':
    char_set = open('chn.txt', 'r', encoding='utf-8').readlines()
    for i in range (0, len(char_set)):
        c = char_set[i].strip('\n')
        encode_dct[c] = i
    #char_set = ''.join([ch.strip('\n') for ch in char_set] + ['卍'])
    char_set = [c.strip('\n') for c in char_set]
    char_set.append('卍')
    nclass = len(char_set)
    K.set_session(get_session())
    #reload(densenet)
    save_model, model = get_model(True, img_h, nclass)
    import sys
    if len(sys.argv) >= 2:
        modelPath = sys.argv[1]
        if os.path.exists(modelPath):
            print("Loading model weights...")
            model.load_weights(modelPath)
            print('done!')
    #train_loader = gen('./output/default/tmp_labels.txt', './output/default/', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    #test_loader = gen('./test/default/tmp_labels.txt', './test/default/', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    train_loader = gen('./output/' + tag + '/tmp_labels.txt', './output/' + tag +'/', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    test_loader = gen('./test/' + tag + '/tmp_labels.txt', './test/' + tag + '/', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    #train_loader = gen('../all/train.txt', '../all/', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    #test_loader = gen('../all/test.txt', '../all/', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    #train_loader = gen('../all/train_13_100.txt', '../all/', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    #test_loader = gen('../all/test_13_100.txt', '../all/', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    checkpoint = ModelCheckpoint(filepath='./tmp_models/weights_' + tag + '_resnet-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=False, save_weights_only=True)
    checkpoint.set_model(save_model)
    #lr_schedule = lambda epoch: 0.0005 * 0.4**epoch
    #lr_schedule = lambda epoch: 0.005 * 20 * 0.4 / (epoch + 1)
    #lr_schedule = lambda epoch: 0.00135 * 2 * 0.33**epoch
    lr_schedule = lambda epoch: 0.0005 * 1 * 0.55**epoch
    
    learning_rate = np.array([lr_schedule(i) for i in range(30)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    tensorboard = TensorBoard(log_dir='./models/logs', write_graph=True)
    print('-----------Start training-----------')
    model.fit_generator(train_loader,
        steps_per_epoch = train_size // batch_size,
        epochs = 30,
        initial_epoch = 0,
        validation_data = test_loader,
        validation_steps = test_size // batch_size,
        #callbacks = [checkpoint, earlystop, changelr, tensorboard])
        #callbacks = [checkpoint, changelr, tensorboard])
        callbacks = [checkpoint, tensorboard])

