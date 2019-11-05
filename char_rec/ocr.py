# -*- coding:utf-8 -*-
import os
import sys
import cv2
import time
import copy
import logging
from math import *
import numpy as np
from PIL import Image
from char_rec.predict import predict

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
                        chn_model_path = './char_rec/models/weights_chn_1028_shufflenet_chage_lr01-avg_1+2+3.h5',
                        jap_model_path = './char_rec/models/weights_jap_1101_shufflenet_change_lr01-avg1+2+3.h5',
                        chn_res_model_path = './char_rec/models/weights_chn_0925_resnet-05-one.h5')

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

def charRec(lan, img, text_recs, angle):
    '''
    lan:语言参数CHE中英；JPE英日；ENG纯英
    img:图片
    text_recs：box
    angle：True、False是否需要角度
    '''
    # if lan.upper() == 'CHE':
    #     print('CHE')
    #     keras_densenet = keras_densenet_ch
    # elif lan.upper() == 'JPE':
    #     print('JPE')
    #     keras_densenet = keras_densenet_jp
    # elif lan.upper() == 'ENG':
    #     print('ENG')
    #     keras_densenet = keras_densenet_eng
    # else:
    #     print('CHE')
    #     keras_densenet = keras_densenet_ch
    t0 = time.time()
    xDim, yDim = img.shape[1], img.shape[0]
    h, w = img.shape[:2]
    #print('angle',angle)
    #print('lan',lan)
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
    # print(rec_trans.shape)
    # print(len(text_recs))

    results = []
    image_info = []
    for i in range(len(text_recs)):
        r = [int(a) for a in rec_trans[i]]
        # print(rec_trans[i])
        # print(r)
        # print(img_trans.shape)
        he = r[5] - r[1]
        if he > 50:
            y_offset = 4
        else:
            y_offset = 2        
        partImg = img_trans[max(1, r[1]-y_offset):min(h, r[5]+y_offset), max(1, r[0]-2):min(w, r[2]+2)]
        # print(partImg.shape)
        # cv2.imwrite('re.jpg', img_trans)
        # if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > 2 * partImg.shape[1]:
        if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > 2 * partImg.shape[1]:
            results.append({'location':[int(i) for i in text_recs[i]], 'text': '', 'scores':[0.0]})
            continue
        #part_img_tmp_path  = './char_rec/tmp_rec/'
        #picname = str(time.time()) + '.jpg'
        #cv2.imwrite(part_img_tmp_path+picname,partImg)
        image = Image.fromarray(partImg).convert('L')
        #image.save('image_rect/rect_result_'+str(time.time())+'.jpg')
        pic_info = {}
        #pic_info['picname'] = str(picname)
        pic_info['location'] = [int(a) for a in text_recs[i]]
        #logging.info('排序前')
        #logging.info(pic_info['location'])
        width, height = image.size[0], image.size[1]
        scale = height * 1.0 / 32
        width = int(width / scale)
        image = image.resize((width, 32), Image.ANTIALIAS)
        pic_info['image'] = np.array(image)
        image_info.append(pic_info)
    t1 = time.time()
    image_info = sort_box(image_info)
    logging.info('排序时间：%s' %str(time.time() - t1))
    logging.info('预处理时间: %s' %str(time.time() - t0))
    #print('检测框数量',len(image_info))
    batch_image = []
    batch_image_info = []
    if len(image_info) > 0:
        width = image_info[0]['image'].shape[1]
        for index,image1 in enumerate(image_info):
            #print(image)
            #logging.info('排序后')
            #logging.info(image1['location'])
            image = image1['image']

            batch_fill = False

            max_width = width
            min_width = width - 20
            #if batch_fill == False:

            if image.shape[1] >min_width:

                channel_one = np.pad(image, ((0, 0), (0, width - image.shape[1])), 'constant', constant_values=(255, 255))
                img = np.array(channel_one, 'f') / 255.0 - 0.5
                img = np.expand_dims(img, axis=2).swapaxes(0, 1)
                batch_image.append(img)
                batch_image_info.append(image1)
                if index == len(image_info)-1:
                    batch_results = predict.predict_batch(np.array(batch_image),batch_image_info,lan)
                    results +=batch_results
            else:
                #batch_fill = True
                batch_results = predict.predict_batch(np.array(batch_image),batch_image_info,lan)
                results +=batch_results
                #try:
                if True:
                    width = image_info[index]['image'].shape[1]
                    channel_one = np.pad(image, ((0, 0), (0, width - image.shape[1])), 'constant', constant_values=(255, 255))
                    img = np.array(channel_one, 'f') / 255.0 - 0.5
                    img = np.expand_dims(img, axis=2).swapaxes(0, 1)
                    batch_image = [img]
                    batch_image_info = [image1]
                    if index == len(image_info)-1:
                        batch_results = predict.predict_batch(np.array(batch_image),batch_image_info,lan)
                        results +=batch_results
                #except Exception as e :


        '''
        # 对于竖文本进行切分
        if partImg.shape[0] > 2 * partImg.shape[1]:
            img_copy = copy.deepcopy(np.array(image)) 
            part_points = split_str(img_copy)
            points_num = len(part_points)
            if points_num > 2:
                for i in range(0, points_num-1):
                    part_img = img_copy[part_points[i]:part_points[i+1], :]
                    if part_img.shape[0] > 2 * part_img.shape[1]:
                        continue
                    else:
                        part_img = Image.fromarray(part_img)
                        text, scores = keras_densenet(part_img)
                        if len(scores) == 0:
                            continue
                        avg_score = sum([float(j) for j in scores])/len(scores)
                        box = copy.deepcopy([int(j) for j in text_recs[i]])
                        # box = [int(j) for j in text_recs[i]]
                        logging.info(str(box))
                        box[1] += part_points[i]
                        box[3] += part_points[i]
                        box[5] += part_points[i+1]
                        box[7] += part_points[i+1]
                        logging.info(str(box))

                        if len(text) > 0 and avg_score>0.6:
                            results.append({'location': box, 'text': text, 'scores': scores})  # 识别
                        else:
                            results.append({'location': box, 'text': '', 'scores': [0.0]})  # 识别文字 
            continue 
        '''        
            
    logging.info('predict 耗时：%s' %str(predict.predict_time))
    predict.predict_time = 0
    logging.info('decode 耗时：%s' %str(predict.decode_time))
    #predict.decode_time = 0
    logging.info('resnet predict 耗时：%s' %str(predict.res_predict_time))
    predict.res_predict_time = 0
    return results




