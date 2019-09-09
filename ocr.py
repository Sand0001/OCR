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

from psenet.predict import predict as pse
from char_rec.ocr import charRec as rec
# from resnet.ocr import charRec as rec

def sort_box(box):
    """ 
    对box进行排序
    """
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box

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

def draw_box(img, boxes):
    for box in boxes:
        box = [int(i) for i in box]
       # box = box['location']
        cv2.line(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.line(img, (box[2], box[3]), (box[6], box[7]), (255, 0, 0), 2)
        cv2.line(img, (box[0], box[1]), (box[4], box[5]), (255, 0, 0), 2)
        cv2.line(img, (box[4], box[5]), (box[6], box[7]), (255, 0, 0), 2)
 
    cv2.imwrite('images/result'+'_'+str(time.time())+'.jpg', img)

def model(img, lan, angle=False, combine=False, lines=[]):
    """
    @img: 图片
    @adjust: 是否调整文字识别结果
    """
    h, w, _ = img.shape
    #cv2.imwrite('result.jpg', img)
    a = time.time()
    # print(type(angle))
    # print(angle)
    text_recs = pse(img, angle,combine, lines)
    b = time.time()
    logging.info('pse的耗时：%s' % str(b-a))
    results = rec(lan, img, text_recs, angle)
    #logging.info('全部返回的result')
    #logging.info(results)

    # draw_box(img, text_recs)
    # cv2.imwrite('result'+str(time.time())+'.jpg', img)
    c = time.time()
    logging.info('识别的耗时：%s' %str(c-b))
   
    # for i in range(len(text_recs)):
    #     r = [int(a) for a in text_recs[i]]
    #     part_img = img[r[1]:r[5], r[0]:r[2]]        
    #     cv2.imwrite('image_rect/'+str(time.time())+'.jpg', part_img)
    #results = []
    #for i in text_recs:
    #    i = i.tolist()
    #    result = char_rec(img, i)
    #    if result:
    #        results.append(result)
    #    else:
    #        results.append({'location':change_box(i), 'text':''})

    return results, (w, h)
