# -*- coding:utf-8 -*-
import cv2
import time
import logging
from math import *
import numpy as np
from char_rec.predict import predict
from char_rec.get_part_img import get_part_img
from char_rec.utils import dict_add,img_resize,is_valid

'''
predict = predict(chn_charset_path ='./char_rec/corpus/chn.txt',
			eng_charset_path='./char_rec/corpus/eng_new.txt',
                        jap_charset_path='./char_rec/corpus/japeng_new1.txt',
			eng_model_path = './char_rec/models/weights_eng_902_change_symbol_ratio_avg16+17+18.h5',
			chn_model_path = './char_rec/models/weights_chn_0925_resnet-05-one.h5',
			jap_model_path = './char_rec/models/weights_jap_add_fonts1015_avg5+6+7.h5')
'''

predict = predict(chn_charset_path ='./char_rec/corpus/chn.txt',
                        eng_charset_path='./char_rec/corpus/eng_new.txt',
                        jap_charset_path='./char_rec/corpus/japeng_new1.txt',
                        eng_model_path = './char_rec/models/weights_eng_add_fonts1018_shufflenet_chage_lr2-avg2+3+4.h5',
                        chn_model_path = './char_rec/models/weights_chn_1103_seal_bg_fg_shufflenet_change_lr01-02-one.h5',
                        jap_model_path = './char_rec/models/weights_jap_1101_shufflenet_change_lr01-avg1+2+3.h5',
                        chn_res_model_path = './char_rec/models/weights_chn_0925_resnet-05-one.h5')

class INFO():
    def __init__(self):
        self.gen_batch_time = 0
INFO = INFO()
def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])) : min(ydim - 1, int(pt3[1])), max(1, int(pt1[0])) : min(xdim - 1, int(pt3[0]))]
    return imgOut

def sort_box(box):
    """
    对box进行排序
    """
    box = sorted(box, key=lambda x: x['image'].shape[1],reverse= True)
    return box

def warpAffinePadded(src_h,src_w,M,mode='matrix'):
    '''
    重新计算旋转矩阵，防止cut off image
    args：
        src_h,src_w 原图的高、宽
        mode: mode is matrix 时 M 是旋转矩阵
              mode is angle 时  M 是角度
    returns:
        offset_M : 新的旋转矩阵
        padded_w,padded_h : 图像的新宽、高

    ------------------------------------
    用法：
        h,w = imagetest.shape[0:2]
        M = cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
        offset_M , padded_w , padded_h = warpAffinePadded(h,w,M)
        rects = cv2.transform(rects,offset_M)
        imagetest = cv2.warpAffine(imagetest,offset_M,(padded_w,padded_h))
    '''
    if(mode == 'angle'):
        M = cv2.getRotationMatrix2D((src_w/2,src_h/2),M,1.0)

    # 图像四个顶点
    lin_pts = np.array([
        [0,0],
        [src_w,0],
        [src_w,src_h],
        [0,src_h]
    ])
    trans_lin_pts = cv2.transform(np.array([lin_pts]),M)[0]

    #最大最小点
    min_x = np.floor(np.min(trans_lin_pts[:,0])).astype(int)
    min_y = np.floor(np.min(trans_lin_pts[:,1])).astype(int)
    max_x = np.ceil(np.max(trans_lin_pts[:,0])).astype(int)
    max_y = np.ceil(np.max(trans_lin_pts[:,1])).astype(int)

    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0
    #print('offsetx:{},offsety:{}'.format(offset_x,offset_y))
    offset_M = M + [[0,0,offset_x],[0,0,offset_y]]

    padded_w = src_w + (max_x - src_w)  + offset_x
    padded_h = src_h + (max_y - src_h)  + offset_y
    return offset_M,padded_w,padded_h



def gen_batch_predict(image_info,lan):
    gen_time = time.time()
    results = []
    erro_record = {'wrong':0,'all':0}
    batch_image = []
    batch_image_info = []
    image_info_length = len(image_info)
    if image_info_length > 0:
        image_info_b = image_info.copy()
        width = image_info[0]['image'].shape[1]
        for index, image1 in enumerate(image_info):
            image = image1['image']
            min_width = width - 20
            if image.shape[1] > min_width:
                channel_one = np.pad(image, ((0, 0), (0, width - image.shape[1])), 'constant',
                                     constant_values=(255, 255))
                img = np.array(channel_one, 'f') / 255.0 - 0.5
                img = np.expand_dims(img, axis=2).swapaxes(0, 1)
                batch_image.append(img)
                batch_image_info.append([image1])
                image_info_b[index] = 0
            else:
                batch_tmp = [image_info[index]['image'].shape[1]]   # 放的是  宽
                batch_image_info_tmp = [image_info[index]]
                width_list_tmp = [index]    #放的是index
                for j in range(1, image_info_length):
                    width_t = image_info[-j]['image'].shape[1]
                    width_t_1 = image_info[-j-1]['image'].shape[1]
                    sum_batch_tmp = sum(batch_tmp)
                    if min_width <= (sum_batch_tmp+2*len(batch_tmp)+ width_t) <= width and image_info_length - j> index and image_info_b[-j]!=0 :
                        width_list_tmp.append(image_info_length-j)
                        batch_image_info_tmp.append(image_info[image_info_length-j])
                        batch_tmp.append(width_t)
                        image_info_b[-j] = 0
                        image_info_b[index] = 0
                    if (sum(batch_tmp)+2*(len(batch_tmp)+1)+ width_t_1) > width:
                        break
                if len(batch_tmp)!= 1:
                    for ii in range(len(width_list_tmp)-1):
                        if ii == 0:
                            img_1 = np.pad(image_info[width_list_tmp[ii]]['image'], ((0, 0), (0,2)), 'constant',
                                     constant_values=(255, 255))   #做2像素的padding
                        else:
                            img_1 = np.pad(img_1, ((0, 0), (0,2)), 'constant',
                                     constant_values=(255, 255))   #做2像素的padding
                        img_1 = np.concatenate((img_1,image_info[width_list_tmp[ii+1]]['image']),axis=1)
                    channel_one = np.pad(img_1, ((0, 0), (0, width - img_1.shape[1])), 'constant',
                                         constant_values=(255, 255))
                    #cv2.imwrite('/data/fengjing/ocr_recognition_test/html/image_rec/a.jpg',channel_one)
                    img = np.array(channel_one, 'f') / 255.0 - 0.5
                    img = np.expand_dims(img, axis=2).swapaxes(0, 1)
                    batch_image.append(img)
                    batch_image_info.append(batch_image_info_tmp)
        INFO.gen_batch_time += time.time() - gen_time
        batch_results,batch_erro_record = predict.predict_batch_v2(np.array(batch_image), batch_image_info, lan)
        results += batch_results
        erro_record = dict_add(batch_erro_record, erro_record)   #dict 合并相加
        image_info_a = list(filter(None, image_info_b))
        if image_info_a != []:
            batch_results,batch_erro_record = gen_batch_predict(image_info_a,lan)          # 递归调用
            results += batch_results
            erro_record = dict_add(batch_erro_record,erro_record)
        return results,erro_record


def get_image_info(text_recs,rec_trans,img_trans):
    h,w = img_trans.shape[:2]
    image_info = []
    results = []
    for i in range(len(text_recs)):
        r = [int(a) for a in rec_trans[i]]
        he = r[5] - r[1]
        if he > 50:
            y_offset = 4
        else:
            y_offset = 2
        partImg = img_trans[max(1, r[1] - y_offset):min(h, r[5] + y_offset), max(1, r[0] - 2):min(w, r[2] + 2)]
        # if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > 2 * partImg.shape[1]:
        if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > 3 * partImg.shape[1]:
            results.append({'location': [int(i) for i in text_recs[i]], 'text': '', 'scores': [0.0]})
            continue

        pic_info = {}
        pic_info['location'] = [int(a) for a in text_recs[i]]
        # logging.info('排序前')
        # logging.info(pic_info['location'])
        image = img_resize(partImg)
        pic_info['image'] = image
        image_info.append(pic_info)
    return image_info,results

def charRec(lan, img, text_recs, angle):
    '''
    lan:语言参数CHE中英；JPE英日；ENG纯英
    img:图片
    text_recs：box
    angle：True、False是否需要角度
    '''
    t0 = time.time()
    xDim, yDim = img.shape[1], img.shape[0]
    h, w = img.shape[:2]
    if angle:
        angle = text_recs[0][-1]
        rec = np.array(text_recs)[:,:-1].reshape(-1, 4, 2)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        offset_M , padded_w , padded_h = warpAffinePadded(h,w,M)
        rec_trans = cv2.transform(rec,offset_M)
        img_trans = cv2.warpAffine(img,offset_M,(padded_w,padded_h))
        rec_trans = rec_trans.reshape(-1, 8)
    else:
        img_trans = img
        rec_trans = text_recs
    # 使用预处理返回图片信息
    t0_1 = time.time()
    image_info  = get_part_img.get_image_info_with_pre_post(text_recs, rec_trans, img_trans)  # 如果不符合的图片直接不返回了
    t1 = time.time()
    image_info = sort_box(image_info)
    logging.info('获取partimg 时间：%s' %str(t1-t0_1))
    logging.info('排序时间：%s' %str(time.time() - t1))
    logging.info('预处理时间: %s' %str(time.time() - t0))
    logging.info('gen batch 时间: %s' % str(INFO.gen_batch_time))
    INFO.gen_batch_time = 0
    logging.info('检测框数量  %s' %str(len(image_info)))
    results,erro_record = gen_batch_predict(image_info, lan)
    logging.info('返回结果数量  %s' % str(len(results)))
    #logging.info('img is valid ? %s' %str(is_valid(erro_record)))
    #print('img is valid ?', is_valid(erro_record))

    logging.info('predict 耗时：%s' %str(predict.predict_time))
    predict.predict_time = 0
    logging.info('decode 耗时：%s' %str(predict.decode_time))
    predict.decode_time = 0

    return results if is_valid(erro_record) else []



