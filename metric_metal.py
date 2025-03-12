import numpy as np
import cv2
import os
import shutil
import matplotlib.pyplot as plt

pred_dir1 = 'E:/selfsupervised/mocoPreheat/fine_UNet_typical_moco_tmp05_mdr07/_Num1/1_0/epoch_199'
pred_dir2 = 'E:/selfsupervised/mocoPreheat/fine_UNet_typical_simclr_tmp05_mdr07/_Num1/1_0/epoch_199'
pred_dir1_5 = 'E:/selfsupervised/mocoPreheat/fine_UNet_typical_moco_tmp05/_Num5/5_2/epoch_199'
pred_dir2_5 = 'E:/selfsupervised/mocoPreheat/fine_UNet_typical_simclr_tmp05/_Num5/5_1/epoch_399'
pred_dir1_f = 'E:/selfsupervised/mocoPreheat/fine_UNet_typical_moco_tmp05/_Numf/f/epoch_49'
pred_dir2_f = 'E:/selfsupervised/mocoPreheat/fine_UNet_typical_simclr_tmp05/_Numf/f/epoch_49'

gt_dir = 'E:/selfsupervised/dataset/preheat/test_numf/gt'
values = [64, 128, 255]
gt_values = [1, 2, 3]
keys = dict()
keys[64] = 'c2'
keys[128] = 'c1'
keys[255] = 'c3'
gt_keys = dict()
gt_keys[1] = 'c1'
gt_keys[2] = 'c2'
gt_keys[3] = 'c3'
temps = ['15-30','45-60','90-120']
min_area = 10

def inst_growing(img, values, keys,
                 c1_seedMarks, c1_nums, c1_hs, c1_ws, c1_sizes,
                 c2_seedMarks, c2_nums, c2_hs, c2_ws, c2_sizes,
                 c3_seedMarks, c3_nums, c3_hs, c3_ws, c3_sizes):
    class Point(object):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def getX(self):
            return self.x

        def getY(self):
            return self.y
    def getGrayDiff(img, currentPoint, tmpPoint):
        return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))
    def selectConnects():
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1),
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]  # 八邻域
        return connects
    def regionGrow(img, seeds, seedMark, label, thresh=1):
        height, weight = img.shape
        seedList = []
        for seed in seeds:
            seedList.append(seed)
        connects = selectConnects()
        while (len(seedList) > 0):
            currentPoint = seedList.pop(0)  # 弹出第一个元素
            seedMark[currentPoint.x, currentPoint.y] = label
            for i in range(8):
                tmpX = currentPoint.x + connects[i].x
                tmpY = currentPoint.y + connects[i].y
                if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                    continue
                grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
                if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                    seedMark[tmpX, tmpY] = label
                    seedList.append(Point(tmpX, tmpY))
        return seedMark
    def getmetric(seedmark):
        num_ = 0
        h_ = []
        w_ = []
        size_ = []
        for i in range(int(np.max(seedmark))):
            if i == 0 or len(np.where(seedmark == i)[0]) == 0:
                continue
            num_ += 1
            xs, ys = np.where(seedmark == i)
            size_.append(len(xs))
            xs_len = np.max(xs) - np.min(xs) + 1
            ys_len = np.max(ys) - np.min(ys) + 1
            if xs_len > ys_len:
                h_.append(xs_len)
                w_.append(ys_len)
            else:
                h_.append(ys_len)
                w_.append(xs_len)
        h_ = np.array(h_)
        w_ = np.array(w_)
        size_ = np.array(size_)
        #return num_, (np.mean(h_) + np.mean(w_)) / 2, np.mean(h_ / w_), np.mean(size_)
        return [num_], h_, w_, size_

    for value in values:
        seedMark = np.zeros(img.shape)
        label = 0
        seedFull = np.zeros(img.shape)
        seedFull[np.where(img == value)] = 1
        while(len(np.where(seedMark>0)[0]) != len(np.where(seedFull==1)[0])):
            label += 1
            xs, ys = np.where(np.logical_and(seedMark ==0, seedFull==1) == True)
            seeds = [Point(xs[0],ys[0])]
            seedMark = regionGrow(img, seeds, seedMark, label)
        if keys[value] == 'c1':
            c1_seedMarks.append(seedMark)
            nums_, h_, w_, size_ = getmetric(seedMark)
            c1_nums.extend(nums_)
            c1_hs.extend(h_)
            c1_ws.extend(w_)
            c1_sizes.extend(size_)
        elif keys[value] == 'c2':
            c2_seedMarks.append(seedMark)
            nums_, h_, w_, size_ = getmetric(seedMark)
            c2_nums.extend(nums_)
            c2_hs.extend(h_)
            c2_ws.extend(w_)
            c2_sizes.extend(size_)
        elif keys[value] == 'c3':
            c3_seedMarks.append(seedMark)
            nums_, h_, w_, size_ = getmetric(seedMark)
            c3_nums.extend(nums_)
            c3_hs.extend(h_)
            c3_ws.extend(w_)
            c3_sizes.extend(size_)

    return c1_seedMarks, c1_nums, c1_hs, c1_ws, c1_sizes, \
        c2_seedMarks, c2_nums, c2_hs, c2_ws, c2_sizes, \
        c3_seedMarks, c3_nums, c3_hs, c3_ws, c3_sizes

def eval(dirname, values, keys):
    print(dirname)
    for temp in temps:
        c1_seedMarks, c1_nums, c1_hs, c1_ws, c1_sizes = [], [], [], [], []
        c2_seedMarks, c2_nums, c2_hs, c2_ws, c2_sizes = [], [], [], [], []
        c3_seedMarks, c3_nums, c3_hs, c3_ws, c3_sizes = [], [], [], [], []
        for file in os.listdir(dirname):
            if file.split('_')[0] in temp:
                img = cv2.imread(dirname + '/' + file, 0)
                c1_seedMarks, c1_nums, c1_hs, c1_ws, c1_sizes, \
                c2_seedMarks, c2_nums, c2_hs, c2_ws, c2_sizes, \
                c3_seedMarks, c3_nums, c3_hs, c3_ws, c3_sizes = inst_growing(img, values, keys,
                                                                      c1_seedMarks, c1_nums, c1_hs, c1_ws, c1_sizes,
                                                                      c2_seedMarks, c2_nums, c2_hs, c2_ws, c2_sizes,
                                                                      c3_seedMarks, c3_nums, c3_hs, c3_ws, c3_sizes)
        c1_nums, c1_hs, c1_ws, c1_sizes = np.array(c1_nums), np.array(c1_hs), np.array(c1_ws), np.array(c1_sizes)
        c2_nums, c2_hs, c2_ws, c2_sizes = np.array(c2_nums), np.array(c2_hs), np.array(c2_ws), np.array(c2_sizes)
        c3_nums, c3_hs, c3_ws, c3_sizes = np.array(c3_nums), np.array(c3_hs), np.array(c3_ws), np.array(c3_sizes)

        c1_idx = np.where(c1_sizes<min_area)
        c2_idx = np.where(c2_sizes<min_area)
        c3_idx = np.where(c3_sizes<min_area)

        c1_hs, c1_ws, c1_sizes = np.delete(c1_hs, c1_idx), np.delete(c1_ws, c1_idx), np.delete(c1_sizes, c1_idx)
        c2_hs, c2_ws, c2_sizes = np.delete(c1_hs, c2_idx), np.delete(c1_ws, c2_idx), np.delete(c1_sizes, c2_idx)
        c3_hs, c3_ws, c3_sizes = np.delete(c1_hs, c3_idx), np.delete(c1_ws, c3_idx), np.delete(c1_sizes, c3_idx)

        temp_c1_nums, temp_c2_nums, temp_c3_nums = np.mean(c1_nums)-len(c1_idx)/len(c1_nums), np.mean(c2_nums)-len(c2_idx)/len(c2_nums), np.mean(c3_nums)-len(c3_idx)/len(c3_nums)
        temp_c1_lens, temp_c2_lens, temp_c3_lens = np.mean(np.sqrt(c1_hs**2+c1_ws**2)), np.mean(np.sqrt(c2_hs**2+c2_ws**2)), np.mean(np.sqrt(c3_hs**2+c3_ws**2))
        temp_c1_size, temp_c2_size, temp_c3_size = np.mean(c1_sizes), np.mean(c2_sizes), np.mean(c3_sizes)
        print('[temp]:' + temp)
        print('[class]:' + 'c1' + ' [nums]: {:.2f}'.format(np.mean(temp_c1_nums)) + ' [lens]: {:.2f}'.format(np.mean(temp_c1_lens))+ ' [sizes]: {:.2f}'.format(np.mean(temp_c1_size)))
        #print('[class]:' + 'c2' + ' [nums]: {:.2f}'.format(np.mean(temp_c2_nums)) + ' [lens]: {:.2f}'.format(np.mean(temp_c2_lens))+ ' [sizes]: {:.2f}'.format(np.mean(temp_c2_size)))
        #print('[class]:' + 'c3' + ' [nums]: {:.2f}'.format(np.mean(temp_c3_nums)) + ' [lens]: {:.2f}'.format(np.mean(temp_c3_lens))+ ' [sizes]: {:.2f}'.format(np.mean(temp_c3_size)))



if __name__ == '__main__':
    eval(gt_dir, gt_values, gt_keys)
    eval(pred_dir1, values, keys)
    eval(pred_dir2, values, keys)
    #eval(pred_dir1_5, values, keys)
    #eval(pred_dir2_5, values, keys)
    #eval(pred_dir1_f, values, keys)
    #eval(pred_dir2_f, values, keys)
