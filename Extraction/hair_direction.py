import cv2
import numpy as np
import glob
import math
import matplotlib.pyplot as plt
import sys
# from skimage import filters

PHI = 3.1415926
NaN = float('nan')

filter_num = 32
all_theta = np.arange(0, PHI * 2, PHI * 2 / filter_num)


def display_im(im, name, cmap=''):
    plt.figure()
    if cmap is '':
        plt.imshow(im)
    else:
        plt.imshow(im, cmap=cmap)
    plt.title(name)
    plt.axis('off')

def calc_orientation(im, mask):
    # gabor filter
    kernel_size = (7, 7)
    filtered = np.zeros((hair.shape[0], hair.shape[1], filter_num))
    kernel = [0] * filter_num
    for (idx, theta) in zip(range(0, filter_num), all_theta):
        kernel[idx] = cv2.getGaborKernel(ksize=kernel_size, sigma=1.8, theta=theta, lambd=4,
                                         gamma=1.8 / 2.4)  # , psi=0) # sin
        # print('kernel shape', kernel)
        filtered[:, :, idx] = cv2.filter2D(im, ddepth=-1, kernel=kernel[idx])#ddepth 表示目标图像深度，ddepth=-1 表示生成与原图像深度相同的图像

    # normalize filtered
    denominator = np.sum(filtered * filtered, axis=2, keepdims=True)
    filtered_ = filtered / (np.sqrt(denominator) + 1e-9)

    # calc orientation map
    orientation_init = (filtered_).argmax(axis=2) + 1  # save the theta index, which ranges from 1 to filter_num
    orientation_init[mask == 0] = 0

    # transform to tangent angle
    tangent_angle_init = all_theta[orientation_init - 1] + PHI / 2
    tangent_angle = tangent_angle_init.copy()
    tangent_angle[np.where((tangent_angle >= PHI) & (tangent_angle < PHI * 2))] = tangent_angle[np.where(
        (tangent_angle >= PHI) & (tangent_angle < PHI * 2))] - PHI
    tangent_angle[np.where((tangent_angle >= 2 * PHI))] = tangent_angle[np.where((tangent_angle >= 2 * PHI))] - PHI * 2


    display_im(tangent_angle, 'tangent angle', cmap=plt.cm.jet)

    return tangent_angle

def LBP(image):
    W, H = image.shape                    #获得图像长宽
    xx = [-1,  0,  1, 1, 1, 0, -1, -1]
    
    yy = [-1, -1, -1, 0, 1, 1,  1,  0]    #xx, yy 主要作用对应顺时针旋转时,相对中点的相对值.
    res = np.zeros((W - 2, H - 2),dtype="uint8")  #创建0数组,显而易见维度原始图像的长宽分别减去2，并且类型一定的是uint8,无符号8位,opencv图片的存储格式.
    for i in range(1, W - 2):
        for j in range(1, H - 2):
            temp = ""
            for m in range(8):
                Xtemp = xx[m] + i    
                Ytemp = yy[m] + j    #分别获得对应坐标点
                if image[Xtemp, Ytemp] > image[i, j]: #像素比较
                    temp = temp + '1'
                else:
                    temp = temp + '0'
            res[i - 1][j - 1] =int(temp, 2)   #写入结果中
    display_im(res,'LBP')
    return res

def LBP_circle(image):
    W, H = image.shape                    #获得图像长宽
    xx = [0,  1,  2, 1, 0, -1, -2, -1]
    
    yy = [-2, -1, 0, 1, 2, 1,  0,  -1]    #xx, yy 主要作用对应顺时针旋转时,相对中点的相对值.
    res = np.zeros((W - 3, H - 3),dtype="uint8")  #创建0数组,显而易见维度原始图像的长宽分别减去2，并且类型一定的是uint8,无符号8位,opencv图片的存储格式.
    for i in range(2, W - 3):
        for j in range(2, H - 3):
            temp = ""
            for m in range(8):
                Xtemp = xx[m] + i    
                Ytemp = yy[m] + j    #分别获得对应坐标点
                if image[Xtemp, Ytemp] > image[i, j]: #像素比较
                    temp = temp + '1'
                else:
                    temp = temp + '0'
            res[i - 1][j - 1] =int(temp, 2)   #写入结果中
    display_im(res,'LBP_circle')
    return res

if __name__ == '__main__':
    
    im = cv2.imread('/Users/xing/Y-lab/ori_bgr.jpg')
    mask = cv2.imread('/Users/xing/Y-lab/hair_mask.jpg')
    
    b,g,r = cv2.split(mask)
    mask = cv2.merge((r,g,b))
    b,g,r = cv2.split(im)
    im = cv2.merge((r,g,b))
    display_im(im, 'im')
    
    #get hair part
    mask[mask > 50] = 255
    mask[mask <= 50] = 0
    # mask = cv2.ximgproc.dtFilter(guide=im, src=mask, sigmaSpatial=500, sigmaColor=100)
    # # # print(np.unique(mask))
    # thres, mask = cv2.threshold(mask, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

    # get hair image
    mask_pts = np.where(mask == 255)
    pt_row, pt_col = np.array(mask_pts[0]), np.array(mask_pts[1])
    mask_pts = np.concatenate((pt_col[:, np.newaxis], pt_row[:, np.newaxis]), axis=1)
    x, y, w, h = cv2.boundingRect(np.array(mask_pts, dtype='float32'))
    hair = im[y : y + h, x : x + w, :]
    hair[np.where(mask[y : y + h, x : x + w, :] != 255)] = 0

    # plt.imshow(hair.astype(np.uint8))
    display_im(hair.astype(np.uint8), 'hair')
    # gabor filter, get orientation and confidence
    hair_gray = cv2.cvtColor(hair, cv2.COLOR_RGB2GRAY)
    orientation = calc_orientation(hair_gray, mask[y : y + h, x : x + w, 0])
    orientation2 = LBP(hair_gray)
    orientation3 = LBP_circle(hair_gray)
    plt.show()


    # fig, ax = plt.subplots(nrows=2, ncols=2)
    #
    # image = data.coins()
    # edges = filters.sobel(image)
    #
    # low = 0.1
    # high = 0.35
    #
    # lowt = (edges > low).astype(int)
    # hight = (edges > high).astype(int)
    # hyst = filters.apply_hysteresis_threshold(edges, low, high)
    #
    # ax[0, 0].imshow(hyst + hight, cmap='gray')
    # ax[0, 0].set_title('Original image')
    #
    # ax[0, 1].imshow(hight, cmap='magma')
    # ax[0, 1].set_title('Sobel edges')
    #
    # ax[1, 0].imshow(lowt, cmap='magma')
    # ax[1, 0].set_title('Low threshold')
    #
    # ax[1, 1].imshow(hyst, cmap='magma')
    # ax[1, 1].set_title('Hysteresis threshold')
    #
    # print(np.max(hyst + hight))
    #
    # for a in ax.ravel():
    #     a.axis('off')
    #
    # plt.tight_layout()
    #
    # plt.show()