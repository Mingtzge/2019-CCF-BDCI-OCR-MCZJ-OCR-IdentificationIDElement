# -*- coding: utf-8 -*-
# @Time         :  19-11-19  22:43
# @Author       :  Miao Wenqiang
# @Reference    :  None
# @File         :  cut_twist_join.py
# @IDE          :  PyCharm Community Edition
"""
判断身份证正反面方向是否正确，如果方向是反的，则纠正。
需要的参数有：
    1.图片所在路径。
    2.图片处理结果保存路径

输出结果为：
    旋转后的身份证正反面图片。
    对于图片A.jpg,输出A_0.jpg和A_1.jpg,其中A_0.jpg表示身份证反面(国徽面)，A_1.jpg表示身份证正面
"""


import numpy as np
import cv2
import os
from cut_twist_process import cut_part


def compare_a_template(img_gray, template):  # 函数返回模板匹配的最大值
    """
    将图片与模板对比，比较相似度
    :param img_gray: 灰度图片
    :param template: 模板图片
    :return: 相似度，介于[0，1]
    """
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         # 转为灰度图
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)  # 模板匹配
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val     # 返回的是归一化的相似度的最大值,值位于0-1之间


def gaussian_blur(img, name='1.jpg', save_path='./save_temp'):      # 将图像缩放到同一尺寸并进行高斯模糊
    """
    缩放图片并进行高斯模糊
    :param img: 原始图片
    :param name: 图片名，测试用
    :param save_path: 保存路径，测试用
    :return: 统一resize为（160，100）并进行了高斯模糊的图片
    """
    img = cv2.resize(img, (160, 100))     # 这里模板是从图片中随机抽取一张并按照相同方法模糊得来的
    img_blurred = cv2.GaussianBlur(img, ksize=(11, 7), sigmaX=0, sigmaY=0)
    # cv2.imwrite(os.path.join(save_path, name), img_blurred)
    return img_blurred


def flip_one_img(img, name='1.jpg', save_path='./save_temp'):  # 翻转图片
    """
    180度翻转图片
    :param img: 图片
    :param name: 图片名，测试用
    :param save_path: 保存路径，测试用
    :return: 翻转后的图片
    """
    flipped_img = cv2.flip(img, -1)     # 水平,竖直方向翻转图片，180度
    # cv2.imwrite(os.path.join(save_path, name), flipped_img)
    return flipped_img


def judge_img_mode(img_rgb, template_list, norm_pram=[0.95, 0.95, 0.7, 0.7], name='1.jpg'):  # 判断图片所属模式
    """
    判断图片所属的模式，0 反面反  1 反面正  2 正面反   3 正面正
    :param img_rgb: RGB图片
    :param template_list: 模板名组成的列表
    :param norm_pram:  归一化超参数
    :param name: 保存路径，测试用
    :return:
    """
    img = gaussian_blur(img_rgb, name)  # 滤波
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    judge_res = []
    for i in range(4):
        max_p = compare_a_template(img_gray, template_list[i])  # 模板匹配，计算相似度
        # print('the number of template ', i, ' is ', max_p/norm_pram[i])
        judge_res.append(max_p/norm_pram[i])  # 除以一个超参数，平衡不同模板匹配结果值的大小不同，相当于归一化

    return np.argsort(judge_res)[-1], judge_res[np.argsort(judge_res)[-1]]     # 返回四种模式中值最大的那个的索引,及其值


# ################################20191107更新#########################################
def first_judge_one_img(img_rgb, template_list, norm_parm=[0.95, 0.95, 0.7, 0.7], img_name='1.jpg', save_path='./save_imgs/', img_path='./imgs/'):  # 针对新水印作出的改进
    """
    针对复赛新的水印作出的改进，先去判断图片中的水印是否是新水印，
    如果是，则按照此函数处理，不是则用之前的方法处理
    :param img_rgb: RGB图片
    :param template_list: 模板名组成的列表
    :param norm_parm:  归一化参数
    :param img_name:  图片名，测试用
    :param save_path:  保存路径，测试用
    :param img_path: 图片所在路径
    :return:  四个值 flag_judge True 或 False， 表示是否成功处理,
    img_rgb 若flag_judge为True， 表示处理后的RGB图片, 若为False， 无意义
    mode_res 0或1,表示身份证正反面
    mode_value 置信度，表示对mode_res的相信程度，介于[0，1]
    """
    flag_judge = False  # 返回判断，如果为True，则代表已经处理成功，不需要后续操作
    img_rgb1 = cut_part.cut_part_img(img_rgb, 0.08)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    img_gray1 = cv2.cvtColor(img_rgb1, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    zheng_res = compare_a_template(img_gray, template_list[4])  # 新的模板的索引，对应文字正着的图片
    # 这里需要注意的是，必须是原图！不能是裁剪后的，否则可能会出错！
    fan_res = compare_a_template(img_gray, template_list[5])    # 新的模板的索引，对应文字反着的图片

    if zheng_res > 0.4:  # 代表文字是正的，不需要翻转
        flag_judge = True
    elif fan_res > 0.4:
        flag_judge = True
        img_rgb = flip_one_img(img_rgb, img_name)  # 是倒的，需要翻转180度
        img_rgb1 = flip_one_img(img_rgb1, img_name)  # 是倒的，需要翻转180度
        # cv2.imwrite(os.path.join(save_path, img_name), img_rgb)
    else:  # 说明不是新水印，直接返回，后续处理
        return flag_judge, None, None, None

    res = cv2.matchTemplate(img_gray1, template_list[4], cv2.TM_CCOEFF_NORMED)  # 模板匹配
    _, max_val, _, left_top = cv2.minMaxLoc(res)
    #print('max_val', max_val, left_top)
    if left_top[0] > 220:  # 实验发现新水印在右下角容易出错，因此用一个掩模覆盖水印区域
        mask = np.zeros(img_rgb1.shape[:2], dtype='uint8')  # 构建相同大小的掩模模板
        mask[max(0, left_top[1]-30):min(img_rgb1.shape[0], left_top[1] + 120),
                left_top[0]-30:min(img_rgb1.shape[1], left_top[0] + 120)] = 255  # 令水印区域为白色
        mask = 255 - mask  # 减去之后，水印区域为黑色
        mask2 = np.zeros(img_rgb1.shape, dtype='uint8')
        mask2[max(0, left_top[1] - 30):min(img_rgb1.shape[0], left_top[1] + 120),
                left_top[0] - 30:min(img_rgb1.shape[1], left_top[0] + 120)] = 128   # 水印区域为黑色
        masked = cv2.bitwise_and(img_rgb1, img_rgb1, mask=mask)  # 将模板与原图进行位与操作 结束后水印区域为黑色，其余区域不变
        maskednew = cv2.add(masked, mask2)  # 相加，水印区域变为灰色，其余不变

        #cv2.imwrite(os.path.join('./temp/', img_name.split('.')[0]+'_masked.jpg'), masked)
        #cv2.imwrite(os.path.join('./temp/', img_name.split('.')[0] + '_maskednew.jpg'), maskednew) # 测试用

        mode_res, mode_value = judge_img_mode(maskednew, template_list, norm_pram=norm_parm, name=img_name)
    else:  # 其余情况不易出错，直接判断
        mode_res, mode_value = judge_img_mode(img_rgb1, template_list, norm_pram=norm_parm, name=img_name)
    return flag_judge, img_rgb, mode_res//2, mode_value

# ############################20191107更改结束#############################################


def post_judge_one_img(img_rgb, template_list, norm_parm=[0.95, 0.95, 0.7, 0.7], img_name='1.jpg', img_path='./imgs/', save_path='./save_imgs/'):   # 处理一张图片
    """
    初赛的旋转判断程序，所有参数含义同上
    :param img_rgb:
    :param template_list:
    :param norm_parm:
    :param img_name:
    :param img_path:
    :param save_path:
    :return:
    """
    # img_rgb = cv2.imread(os.path.join(img_path, img_name))
    img_rgb1 = cut_part.cut_part_img(img_rgb, 0.08)
    # 这里裁剪是因为输入图片未裁剪，而后续分割有需要未裁剪图片，
    # 但我之前的代码是用裁剪后的图片，因此这里裁剪一下，避免出错
    mode_res, mode_value = judge_img_mode(img_rgb1, template_list, norm_pram=norm_parm, name=img_name)
    if mode_res % 2 == 0:  # 文字是倒的，需要翻转
        img_rgb = flip_one_img(img_rgb, img_name)

    # res_img_name = img_name.split('_')[0] + '_' + str(mode_res//2) + '.jpg'
    # cv2.imwrite(os.path.join(save_path, res_img_name), img_rgb)  # 测试用

    return img_rgb, mode_res//2, mode_value


def preprocess_twist_one_img(img_rgb, template_list, norm_parm=[0.95, 0.95, 0.7, 0.7], img_name='1.jpg', img_path='./imgs/', save_path='./save_imgs/'):
    """
    旋转一张身份证图片，参数含义同上
    :param img_rgb:
    :param template_list:
    :param norm_parm:
    :param img_name:
    :param img_path:
    :param save_path:
    :return:
    """
    flag_judge, img_rgb_res, mode_res, mode_value = first_judge_one_img(img_rgb=img_rgb,
                template_list=template_list, norm_parm=norm_parm, img_name=img_name, save_path=img_path, img_path=save_path)
    if flag_judge == False:  # 如果水印类型是初赛水印或者没有处理成功
        img_rgb_res, mode_res, mode_value = post_judge_one_img(img_rgb=img_rgb,
                template_list=template_list, norm_parm=norm_parm, img_name=img_name, save_path=img_path, img_path=save_path)

    return flag_judge, img_rgb_res, mode_res, mode_value


def judge_img(img_path, save_path, template_names, norm_parm=[0.95, 0.95, 0.7, 0.7]):      # 处理文件夹下所有图片
    """
    处理文件夹下的所有图片
    :param img_path:
    :param save_path:
    :param template_names:
    :param norm_parm:
    :return:
    """
    if not os.path.exists(img_path):  # 判断图片路径是否存在
        print('img path {name} is not exits， program break.'.format(name=img_path))
        return
    if not os.path.exists(save_path):  # 保存路径不存在，则创建路径
        os.makedirs(save_path)

    img_names = os.listdir(img_path)
    img_names.sort()  # 至关重要

    template_list = []
    for template_name in template_names:  # 读取模板
        template_list.append(cv2.imread(template_name, 0))

    for img_name in img_names:
        img_rgb = cv2.imread(os.path.join(img_path, img_name))
        _, img_rgb_res, mode_type, _ = preprocess_twist_one_img(img_rgb, template_list, norm_parm=norm_parm,
                    img_name=img_name, img_path=img_path, save_path=save_path)
        res_img_name = img_name.split('_')[0] + '_' + str(mode_type) + '.jpg'
        cv2.imwrite(os.path.join(save_path, res_img_name), img_rgb_res)


if __name__ == '__main__':
    twist_img_path = 'E:/Python/IDCARD/OCR_CRNN_Residual/test/preprocess_traindata/res_crop/res_fusai_train/'
    twisted_save_path = './res_fusai_train/'
    twist_template_names = ['./template/fan_blurred_fan.jpg',  # 0 反面反
                      './template/fan_blurred_zheng.jpg',  # 1 反面正
                      './template/zheng_blurred_fan.jpg',  # 2 正面反
                      './template/zheng_blurred_zheng.jpg',  # 3 正面正
                      './template/zheng_new.jpg',  # 4 新水印正面
                      './template/fan_new.jpg'  # 5 新水印反面
                      ]
    # 这样定义的四个模式,只需要对2取模值,如果为0则需要翻转;
    # 整除2后, 如果结果为0则代表反面,否则为正面
    twist_norm_prams = [0.95, 0.95, 0.7, 0.7]  # 考虑到匹配结果的误差,定义的归一化参数 [0.95, 0.95, 0.7, 0.7]

    #judge_img(img_path=twist_img_path, save_path=twisted_save_path,
    # template_names=twist_template_names, norm_parm=twist_norm_prams)

