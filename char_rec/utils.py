import cv2
import time

def exc_time(func):
    def warpper(*args, **args2):
        t0 = time.time()
        #print ("@%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        #print ("@%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print ("@%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return warpper

def draw_box(img, boxes):
    for box in boxes:
       # box = box['location']
        box = [int(ele) for ele in box]
        cv2.line(img, (box[0], box[1]-2), (box[2], box[3]), (255, 0, 0), 2)
        cv2.line(img, (box[2], box[3]), (box[6], box[7]), (255, 0, 0), 2)
        cv2.line(img, (box[0], box[1]-2), (box[4], box[5]), (255, 0, 0), 2)
        cv2.line(img, (box[4], box[5]), (box[6], box[7]), (255, 0, 0), 2)
    return img

def img_resize(partImg):
    image = cv2.cvtColor(partImg, cv2.COLOR_BGR2GRAY)
    width, height = image.shape[1], image.shape[0]
    scale = height * 1.0 / 32
    width = int(width / scale)
    image = cv2.resize(image, (width, 32))
    return image

def dict_add(a,b):
    a['wrong'] = a['wrong'] + b['wrong']
    a['all'] = a['all'] +b['all']
    return a

def is_valid(erro_record):
    if erro_record['all'] > 10:

        return False if erro_record['wrong']*1.0/erro_record['all']>0.3 else True
    else:
        return True