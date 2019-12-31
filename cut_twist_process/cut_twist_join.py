# -*- coding: utf-8 -*-
# @Time         :  19-11-19  22:10
# @Author       :  Miao Wenqiang
# @Reference    :  None
# @File         :  cut_twist_join.py
# @IDE          :  PyCharm Community Edition
"""
将身份证正反面从原始图片中切分出来，如果方向不正确则旋转。使用传统图像处理方法实现。
需要的参数有：
    1.图片所在路径。
    2.图片处理结果保存路径

输出结果为：
    切分和旋转后的身份证正反面图片。
    对于图片A.jpg,输出A_0.jpg和A_1.jpg,其中A_0.jpg表示身份证反面(国徽面)，A_1.jpg表示身份证正面
"""


import cv2
import os
from cut_twist_process import cut_part
from cut_twist_process import twist_part


def preprecess_cut_twist_one_img(img_path, img_name, template_list, save_path, norm_parm):
    """
    函数用于处理单张原始图片，切分处身份证正反面并旋转，然后写入到指定目录。
    :param img_path: 图片所在路径
    :param img_name: 图片名
    :param template_list: 保存有模板名的list
    :param save_path: 结果保存路径
    :param norm_parm: 归一化超参数
    :return:  None
    """
    error_temp,  res_bbox = cut_part.preprocess_cut_one_img(img_path, img_name)
    # 裁剪出身份证正反面，error_temp = 0，表示裁剪部分未出错，其余表示出错；res_bbox是身份证区域
    if error_temp == 0:
        flag_judge1, img_rgb_res1, mode_type1, mode_value1 = twist_part.preprocess_twist_one_img(res_bbox[0], template_list, norm_parm=norm_parm)
        # 旋转图片
        flag_judge2, img_rgb_res2, mode_type2, mode_value2 = twist_part.preprocess_twist_one_img(res_bbox[1], template_list, norm_parm=norm_parm)
        if mode_type1 != mode_type2:  # 证明这一对图片处理的没错
            res_img_name1 = img_name.split('.')[0] + '_' + str(mode_type1) + '.jpg'  # 保存时的图片名
            res_img_name2 = img_name.split('.')[0] + '_' + str(mode_type2) + '.jpg'
            cv2.imwrite(os.path.join(save_path, res_img_name1), img_rgb_res1)  # 写入
            cv2.imwrite(os.path.join(save_path, res_img_name2), img_rgb_res2)
        elif mode_value1 > mode_value2:  # 如果识别两个图片同为正面或反面，则判定匹配结果最大的为正确，另一个为另一面
            # print('img {name} was wrong when twist,start correct program'.format(name=img_name))  测试用，打印出错信息
            res_img_name1 = img_name.split('.')[0] + '_' + str(mode_type1) + '.jpg'
            res_img_name2 = img_name.split('.')[0] + '_' + str(abs(1 - mode_type1)) + '.jpg'
            cv2.imwrite(os.path.join(save_path, res_img_name1), img_rgb_res1)
            cv2.imwrite(os.path.join(save_path, res_img_name2), img_rgb_res2)
        else:
            # print('img {name} was wrong when twist,start correct program'.format(name=img_name))
            res_img_name1 = img_name + '_' + str(1 - mode_type2) + '.jpg'
            res_img_name2 = img_name + '_' + str(mode_type2) + '.jpg'
            cv2.imwrite(os.path.join(save_path, res_img_name1), img_rgb_res1)
            cv2.imwrite(os.path.join(save_path, res_img_name2), img_rgb_res2)
    else:  # 处理出错的情况，直接放弃，但要保证处理的结果仍是正反两张图
        img = cv2.imread(os.path.join(img_path, img_name))
        res_img_name1 = img_name + '_0.jpg'
        res_img_name2 = img_name + '_1.jpg'
        img_rgb_res1 = cv2.resize(img, (450, 290))  # 仍缩放到身份证区域大小
        img_rgb_res2 = cv2.resize(img, (490, 290))
        cv2.imwrite(os.path.join(save_path, res_img_name1), img_rgb_res1)
        cv2.imwrite(os.path.join(save_path, res_img_name2), img_rgb_res2)

    return


def process_cut_twist_imgs(img_path, template_names, save_path, norm_parm):
    """
    批量处理目录下的所有原始图片，裁剪出身份证正反面，旋转并保存
    :param img_path: 原始图片所在路径
    :param template_names:  模板地址的列表
    :param save_path: 身份证正反面保存的目录
    :param norm_parm: 归一化参数
    :return: None
    """
    if not os.path.exists(img_path):  # 判断图片路径是否存在
        print('img path {name} is not exits， please check again!'.format(name=img_path))
        return
    if not os.path.exists(save_path):  # 保存路径不存在，则创建路径
        os.makedirs(save_path)

    img_names = os.listdir(img_path) # 列出路径下所有需要处理的图片名
    img_names.sort()  # 排序，至关重要

    template_list = []
    for template_name in template_names:  # 读取模板图片
        template_list.append(cv2.imread(template_name, 0))

    for img_name in img_names:  # 依次处理
        preprecess_cut_twist_one_img(img_path=img_path, img_name=img_name,
                template_list=template_list, save_path=save_path, norm_parm=norm_parm)

    return




if __name__ == '__main__':
    origin_img_path = 'E:/Python/IDCARD/data_fusai/test/'   # 数据集存放路径
    cut_twisted_save_path = './res_fusai_test/'   # 数据集保存路径
    cut_twist_template_names = ['./cut_twist_process/template/fan_blurred_fan.jpg',  # 0 反面反
                      './cut_twist_process/template/fan_blurred_zheng.jpg',  # 1 反面正
                      './cut_twist_process/template/zheng_blurred_fan.jpg',  # 2 正面反
                      './cut_twist_process/template/zheng_blurred_zheng.jpg',  # 3 正面正
                      './cut_twist_process/template/zheng_new.jpg',  # 4 新水印正面
                      './cut_twist_process/template/fan_new.jpg'  # 5 新水印反面
                      ]   # 模板图片路径
    cut_twist_norm_prams = [0.95, 0.95, 0.7, 0.7]  # 超参数，归一化用
    process_cut_twist_imgs(img_path=origin_img_path, template_names=cut_twist_template_names,
            save_path=cut_twisted_save_path, norm_parm=cut_twist_norm_prams)




