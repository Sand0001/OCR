import cv2
import time
import numpy as np
from char_rec.utils import img_resize



def exc_time(func):
    def warpper(*args, **args2):
        t0 = time.time()
        #print ("@%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        #print ("@%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print ("@%.6fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return warpper


class get_part_img():

    def __init__(self):
        self.save_time = 0

    @staticmethod
    def sort_box_by_position_y(box):
        """
        对box进行排序
        """
        box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
        return box

    @staticmethod
    def sort_box_by_position_x(box):
        box = sorted(box, key=lambda x: sum([x[0], x[4]]))
        return box

    @staticmethod
    def extend_pixel(index,box_after_sorted):
        '''

        :param index:
        :param box_after_sorted:
        :param extend_height: 上下的扩充像素
        :return:
        '''

        extend_height = 4
        box = box_after_sorted[index]   # 当前需要判断的box

        compare_index_up = None
        compare_index_down = None   # up and down 都为None 意味着当前列只有他一个  up和 down其中有一个为None 代表在边缘 还是需要判断下的
        for i in range(index+1,len(box_after_sorted)):
            if min(box[2],box_after_sorted[i][2]) - max(box[0],box_after_sorted[i][0]) >= 0:
                compare_index_down = i
                break
        for i in range(index):
            if min(box[2], box_after_sorted[index-1-i][2]) - max(box[0], box_after_sorted[index-1-i][0]) >= 0:
                compare_index_up = index-i-1
                break
        extend_height_final = 0
        for p in range(extend_height):
            if compare_index_down and not compare_index_up:
                if box[5]+extend_height - p < box_after_sorted[compare_index_down][1]:
                    extend_height_final = extend_height - p
                    #print('true',index)
                    break

            elif not compare_index_down and compare_index_up:
                if box[1]- (extend_height -p)  > box_after_sorted[compare_index_up][5]:
                    extend_height_final = extend_height - p
                    break

            elif compare_index_down and compare_index_up:
                if box[1]- (extend_height - p)  >box_after_sorted[compare_index_up][5] and box[5]+extend_height-p < box_after_sorted[compare_index_down][1]:
                    extend_height_final = extend_height - p
                    break

            else:
                extend_height_final = extend_height - p
        return extend_height_final

    @staticmethod
    def get_image_info(partImg,box,img_name = None):
        pic_info = {}
        pic_info['location'] = [int(a) for a in box]
        image = img_resize(partImg)
        pic_info['image'] = image
        if img_name:
            if 'jpg' in img_name:
                picname = img_name.split('.jpg')[0] + '_' + str(time.time()) + '.jpg'
            else:
                picname = img_name.split('.png')[0] + '_' + str(time.time()) + '.jpg'
            pic_info['picname'] = str(picname)
            cv2.imwrite('/data/fengjing/ocr_recognition_test/html/image_rec/' + picname, partImg[:, :, (2, 1, 0)])
        return pic_info

    @staticmethod
    def find_extend_height(max_extend_height,r,img_blank,index):
        for i in range(max_extend_height):
            tmp_extend_height = max_extend_height - i
            up_line = img_blank[r[1] - tmp_extend_height:r[1] - tmp_extend_height +1,r[0]:r[2]]
            down_line = img_blank[r[5]+ tmp_extend_height :r[5] + tmp_extend_height +1,r[0]:r[2]]
            if np.sum(np.array((up_line,down_line))) == 0 :
                return tmp_extend_height
        return 0

    @staticmethod
    def find_extend_width(max_extend_width,r,img_blank,index):
        for j in range(max_extend_width):
            tmp_extend_width = max_extend_width - j
            left_line = img_blank[r[1]:r[5], r[0] - tmp_extend_width:r[0] - tmp_extend_width]
            right_line = img_blank[r[1]:r[5], r[2] + tmp_extend_width:r[2] + tmp_extend_width]

            if np.sum(np.array((left_line,right_line))) == 0:
            # if np.sum(np.hstack((img_blank[r[1]:r[5],
            #           r[0] - tmp_extend_width:r[0]],img_blank[r[1]:r[5],
            #           r[2]:r[2] + tmp_extend_width]))) == 0:
                return tmp_extend_width
        return 0

    @staticmethod
    def get_img_blank(rec_trans,img):
        img_blank = np.zeros(img.shape[:2])
        for index, box in enumerate(rec_trans):
            cv2.rectangle(img_blank, (int(box[0]), int(box[1])), ((int(box[2]), int(box[5]))), color=index)
        return img_blank

    @staticmethod
    def crop_part(img,r,extend_height,extend_width,h,w):
        partImg = img[max(1, r[1] - extend_height):min(h, r[5] + extend_height),  # crop
                  max(1, r[0] - extend_width):min(w, r[2] + extend_width)]
        return partImg

    @staticmethod
    def crop_img(rec_trans, img, picname = None):
        image_info = []
        h, w = img.shape[:2]

        img_blank = get_part_img.get_img_blank(rec_trans,img)
        max_extend_height = 4
        max_extend_width = 3
        for index, box in enumerate(rec_trans):
            r = [int(a) for a in box]
            extend_height = get_part_img.find_extend_height(max_extend_height,r,img_blank,index)
            #extend_height = 1
            extend_width = get_part_img.find_extend_width(max_extend_width, r, img_blank,index)
            #extend_width = 1

            partImg = get_part_img.crop_part(img,r,extend_height,extend_width,h,w)
            if not (partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > 3 * partImg.shape[1]):  # 过滤异常图片
                image_info.append(get_part_img.get_image_info(partImg, r[len(r) // 2:],picname))

        return image_info

    @staticmethod
    def get_image_info_with_pre_post(text_recs,rec_trans,img_trans,picname = None):
        '''

        :param text_recs:  未旋转的坐标
        :param rec_trans:  旋转后的坐标
        :return:
        '''

        rec = np.hstack((np.array(rec_trans),np.array(text_recs)))
        image_info = get_part_img.crop_img(rec, img_trans,picname)

        return image_info


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    from char_rec.utils import draw_box
    img = cv2.imread('/fengjing/test_img/WechatIMG132.jpeg')
    a = np.load('/fengjing/test_img/text_recs.npy')
    plt.figure('1')

    print('框个数',len(a))
    img = draw_box(img,a)
    plt.imshow(img)
    t0 = time.time()
    #get_part_img.crop_img(a, img)
    get_part_img.get_image_info_with_pre_post(a,a,img)
    print('time',time.time()-t0)




