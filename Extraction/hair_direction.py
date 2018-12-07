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

def circular_LBP(src, radius, n_points):
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()
    src.astype(dtype=np.float32)
    dst.astype(dtype=np.float32)

    neighbours = np.zeros((1, n_points), dtype=np.uint8)
    lbp_value = np.zeros((1, n_points), dtype=np.uint8)
    for x in range(radius, width - radius - 1):
        for y in range(radius, height - radius - 1):
            lbp = 0.
            # 先计算共n_points个点对应的像素值，使用双线性插值法
            for n in range(n_points):
                theta = float(2 * np.pi * n) / n_points
                x_n = x + radius * np.cos(theta)
                y_n = y - radius * np.sin(theta)

                # 向下取整
                x1 = int(math.floor(x_n))
                y1 = int(math.floor(y_n))
                # 向上取整
                x2 = int(math.ceil(x_n))
                y2 = int(math.ceil(y_n))

                # 将坐标映射到0-1之间
                tx = np.abs(x - x1)
                ty = np.abs(y - y1)
                
                # f(x,y)=f(0,0)(1-x)(1-y)+f(1,0)x(1-y)+f(0,1)(1-x)y+f(1,1)xy
                # 根据0-1之间的x，y的权重计算公式计算权重
                w1 = (1 - tx) * (1 - ty)
                w2 = tx * (1 - ty)
                w3 = (1 - tx) * ty
                w4 = tx * ty

                # 根据双线性插值公式计算第k个采样点的灰度值
                neighbour = src[y1, x1] * w1 + src[y2, x1] * w2 + src[y1, x2] * w3 + src[y2, x2] * w4

                neighbours[0, n] = neighbour

            center = src[y, x]

            for n in range(n_points):
                if neighbours[0, n] > center:
                    lbp_value[0, n] = 1
                else:
                    lbp_value[0, n] = 0

            for n in range(n_points):
                lbp += lbp_value[0, n] * 2**n

            # 转换到0-255的灰度空间，比如n_points=16位时结果会超出这个范围，对该结果归一化
            dst[y, x] = int(lbp / (2**n_points-1) * 255)
    display_im(dst,'LBP_circle_'+'R='+str(radius)+'_P='+str(n_points))
    return dst

def value_rotation(num):
    value_list = np.zeros((8), np.uint8)
    temp = int(num)
    value_list[0] = temp
    for i in range(7):
        temp = ((temp << 1) | int(temp / 128)) % 256
        value_list[i+1] = temp
    return np.min(value_list)

def rotation_invariant_LBP(src):
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()

    lbp_value = np.zeros((1, 8), dtype=np.uint8)
    neighbours = np.zeros((1, 8), dtype=np.uint8)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]

            center = src[y, x]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                  + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 0] * 128

            # 旋转不变值
            dst[y, x] = value_rotation(lbp)
    display_im(dst,'LBP_invariant')
    return dst

def sharpening(image):
    W, H = image.shape                    #获得图像长宽
    xx = [-1,  0,  1, 1, 1, 0, -1, -1, 0]
    
    yy = [-1, -1, -1, 0, 1, 1,  1,  0, 0]    #xx, yy 主要作用对应顺时针旋转时,相对中点的相对值.
    
    w = [-1,  -1,  -1, -1, -1, -1, -1, -1, 9]
    
    res = np.zeros((W - 2, H - 2),dtype="uint8")  #创建0数组,显而易见维度原始图像的长宽分别减去2，并且类型一定的是uint8,无符号8位,opencv图片的存储格式.
    for i in range(1, W - 2):
        for j in range(1, H - 2):
            temp = 0
            for m in range(9):
                Xtemp = xx[m] + i    
                Ytemp = yy[m] + j    #分别获得对应坐标点
                temp=temp+image[Xtemp, Ytemp]*w[m]
            res[i - 1][j - 1] =temp   #写入结果中
    return res

if __name__ == '__main__':
    
    im = cv2.imread('/Users/xing/Y-lab/ori_bgr.jpg')
    mask = cv2.imread('/Users/xing/Y-lab/hair_mask.jpg')
    
    b,g,r = cv2.split(mask)
    mask = cv2.merge((r,g,b))
    b,g,r = cv2.split(im)
    im = cv2.merge((r,g,b))
    display_im(im, 'im')
    display_im(cv2.merge((sharpening(r),sharpening(g),sharpening(b))),'sharpening')
    
    #get hair part
    mask[mask > 50] = 255
    mask[mask <= 50] = 0
    
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
    orientation = calc_orientation(hair_gray, mask[y : y + h, x : x + w, 0])#gabor filter
    orientation2 = LBP(hair_gray)#LBP
    orientation3 = LBP_circle(hair_gray)#circle LBP without interpolation
    rotation_invariant_LBP(src=hair_gray)#invariant LBP 
    circular_LBP(hair_gray, radius=3, n_points=32)#circle LBP with interpolation
    plt.show()
    
