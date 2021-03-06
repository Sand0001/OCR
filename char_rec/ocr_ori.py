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
from char_rec.densenet_jp.model import predict as keras_densenet_jp
from char_rec.densenet_ch.model import predict as keras_densenet_ch
from char_rec.densenet_eng.model import predict as keras_densenet_eng

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
    if lan.upper() == 'CHE':
        print('CHE')
        keras_densenet = keras_densenet_ch
    elif lan.upper() == 'JPE':
        print('JPE')
        keras_densenet = keras_densenet_jp
    elif lan.upper() == 'ENG':
        print('ENG')
        keras_densenet = keras_densenet_eng
    else:
        print('CHE')
        keras_densenet = keras_densenet_ch
    xDim, yDim = img.shape[1], img.shape[0]
    h, w = img.shape[:2]
    print('angle',angle)
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
        text, scores = keras_densenet(image)
        # logging.info(text)
        if len(scores) == 0:
            results.append({'location':[int(i) for i in text_recs[i]], 'text': '', 'scores':[0.0]})
            continue
        elif len(scores) == 1:
            results.append({'location':[int(i) for i in text_recs[i]], 'text': text,'scores': scores})
            continue
        elif len(text) > 1:
            avg_score = sum([float(i) for i in scores])/len(scores) 
            if avg_score>0.6:
                # logging.info(type(scores))
                results.append({'location': [int(i) for i in text_recs[i]], 'text': text, 'scores': scores})  # 识别
        else:
            results.append({'location': [int(i) for i in text_recs[i]], 'text': '', 'scores': [0.0]})  # 识别文字
            
    return results


