import os
import cv2
import time
import json
import logging
import skimage.io
import numpy as np
from io import BytesIO
from flask_cors import CORS
from flask import Flask, request, make_response

import tensorflow as tf
import keras.backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)
K.set_session(session)

from ocr import model

app = Flask(__name__)
CORS(app, resources=r'/*')

fmt='%(asctime)s | [%(process)d:%(threadName)s:%(thread)d] | [%(filename)s:%(funcName)s: %(lineno)d] | %(levelname)s | %(message)s'
logging.basicConfig(filename='info.log', level=logging.INFO, format=fmt)

@app.route('/ocr_pse_test', methods=['POST'])
def ocr_img_pse():
    file = request.files['file']
    lan = request.args.get('language', 'CHE')
    angle = request.args.get('angle', 'True')
    logging.info('语言类型%s' % lan)
    logging.info('是否需要角度%s' % angle)
    if angle == 'False':
        angle = False
    else:
        angle = True
    if not file:
        logging.info('图片为空')
        return response(json.dumps({'code': -1, 'msg': '文件为空', 'result': ''}))

    #if file.filename.split('.')[-1] not in ['jpg', 'png', 'jpeg']:
     #   logging.info('图片格式错误：%s' % file.filename)
      #  return response(json.dumps({'code': -2, 'msg': '文件格式错误', 'result': ''}))

    logging.info('图片格式：%s' % file.filename)
    # save_path = 'up_images/' + file.filename
    # file.save('mid.jpg')
    # img = cv2.imread(save_path, 3)
    try:
        img_buffer = np.asarray(bytearray(file.read()), dtype='uint8')
        bytesio = BytesIO(img_buffer)
        img = skimage.io.imread(bytesio)
    except Exception as e:
        logging.info(str(e), exc_info=True)
        return response(json.dumps({'code': -2, 'msg': '文件格式错误', 'result': ''}))

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        pass
    a = time.time()
    # cv2.imwrite('mid'+str(a)+'.jpg',img)
    results,img_shape = model(img, lan, angle)
    # logging.info('图片处理完成:'+str(results))
    # return json.dumps({'code':0, 'msg':'', 'result':results, 'shape':img_shape})
    return response(json.dumps({'code':0, 'msg':'', 'result':results, 'shape':img_shape}))

def response(res):
    rst = make_response(res)
    rst.headers['Access-Controlf-Allow-Origin'] = '*'
    rst.headers['Access-Control-Allow-Methods'] = 'POST'
    rst.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return rst

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8019, debug=True, threaded=True)
