# coding=utf-8
import io
import os
import cv2
import time
import json
import logging
import traceback
import skimage.io
import numpy as np
from io import BytesIO
import tornado.ioloop
import tornado.web
from tornado.options import define, options
from tornado.options import options as ops
from tornado import httpclient, gen, ioloop, queues

import tensorflow as tf
import keras.backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
session = tf.Session(config = config)
K.set_session(session)

define("debug",default=True,help="Debug Mode",type=bool)
define("port",default=8007,help="run on this port",type=int)

from ocr import model

fmt='%(asctime)s | [%(process)d:%(threadName)s:%(thread)d] | [%(filename)s:%(funcName)s: %(lineno)d] | %(levelname)s | %(message)s'
logging.basicConfig(filename='ocr-info.log', level=logging.INFO, format=fmt)

class MainHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Headers', 'x-requested-with')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def resp(self, resp_dct):
        resp = json.dumps(resp_dct, ensure_ascii = False)
        # logging.info(resp)
        return resp

    def post(self):
        files = self.request.files.get('file', None)
        lan = self.get_argument('language', 'CHE')
        angle = self.get_argument('angle', 'True')
        combine = self.get_argument('combine', 'False')
        lines = self.get_argument('lines', [])
        just_detection = self.get_argument('just_detection','False')
        if lines:
            lines = eval(lines)
        if not isinstance(lines, list):
            lines = []

        if not files:
            logging.info('图片为空')
            self.write(self.resp({'code':-1, 'msg': '文件为空', 'result': ''}))
            self.finish()
         
        file = files[0]

        logging.info('表格线：%s' % str(lines))
        logging.info('文件%s' % file.filename)
        logging.info('语言类型%s' % lan)
        logging.info('是否需要角度%s' % angle)
        logging.info('是否需要连接%s' % combine)
        logging.info('只做检测%s' % just_detection)

        if angle == 'False':
            angle = False
        else:
            angle = True

        if combine == 'False':
            combine = False
        else:
            combine = True

        just_detection = False if just_detection == 'False' else True

        try:
            img_buffer = np.asarray(bytearray(file.body), dtype='uint8')
            bytesio = BytesIO(img_buffer)
            img = skimage.io.imread(bytesio)
            logging.info('io图片完成!')
        except Exception as e:
            logging.info(str(e), exc_info=True)
            self.write(self.resp({'code': -2, 'msg': '文件格式错误', 'result': ''}))
            self.finish()

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            pass

        start = time.time()
        results, img_shape = model(img, lan, angle, combine, lines ,just_detection)
        end = time.time()
        logging.info('ocr total time %s' % str(end-start))
        #logging.info(results)
        self.write(self.resp({'code':0, 'msg': '', 'result': results, 'shape': img_shape}))

    def options(self):
        # pass
        self.set_status(204)
        self.finish()
        # self.write(self.resp({'code':0, 'msg':''}))


application = tornado.web.Application([
    (r"/ocr_pse_test", MainHandler),
])

if __name__ == '__main__':
    tornado.options.parse_command_line()
    application.listen(ops.port)
    tornado.ioloop.IOLoop.instance().start()
