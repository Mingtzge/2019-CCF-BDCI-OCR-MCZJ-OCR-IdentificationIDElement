#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         :  19-11-19  23:45
# @Author       :  Jiang Mingzhi
# @Reference    :  https://github.com/MaybeShewill-CV/CRNN_Tensorflow
#                  https://github.com/bai-shang/crnn_ctc_ocr.Tensorflow
# @File         :  test_crnn_jmz.py
# @IDE          :  PyCharm Community Edition
"""
识别图片中的文本。需要的参数有：
    1.图片所在路径。
    2.保存有图片名称的txt文件
    3.加载模型的路径

输出结果为：
    csv文件。
"""
import sys
sys.path.append('./')
#print(sys.path)
import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import json

from recognize_process.config import model_config
from recognize_process.crnn_model import crnn_model
from multiprocessing import Pool
CFG = model_config.cfg


def init_args():
    """
    初始化参数
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str,
                        help='Path to the image to be tested',
                        default='./recognize_process/test_imgs/')
    parser.add_argument('-w', '--weights_path', type=str,
                        help='Path to the pre-trained weights to use',
                        default='./recognize_process/model_save/recognize_model')
    parser.add_argument('-c', '--char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored',
                        default='./recognize_process/char_map/char_map.json')
    parser.add_argument('-t', '--txt_path', type=str,
                        help='Whether to display images',
                        default='./recognize_process/anno_test/')

    return parser.parse_args()


def _resize_image(img):
    """
    用于将图片resize为固定高度（32）
    :param img: 输入图片
    :return: resize为固定高度的图片
    """
    dst_height = CFG.ARCH.INPUT_SIZE[1]
    h_old, w_old, _ = img.shape
    height = dst_height
    width = int(w_old * height / h_old)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    return resized_img


def _sparse_matrix_to_list(sparse_matrix, char_map_dict_path=None):
    """
    将矩阵拆分为list，参考：https://github.com/bai-shang/crnn_ctc_ocr.Tensorflow
    :param sparse_matrix:
    :param char_map_dict_path:
    :return:
    """
    indices = sparse_matrix.indices
    values = sparse_matrix.values
    dense_shape = sparse_matrix.dense_shape

    # the last index in sparse_matrix is ctc blanck note
    char_map_dict = json.load(open(char_map_dict_path, 'r',encoding='utf-8'))
    if char_map_dict is None:
        print("error")
    assert (isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')

    dense_matrix = len(char_map_dict.keys()) * np.ones(dense_shape, dtype=np.int32)
    for i, indice in enumerate(indices):
        dense_matrix[indice[0], indice[1]] = values[i]
    string_list = []
    for row in dense_matrix:
        string = []
        for val in row:
            string.append(_int_to_string(val, char_map_dict))
        string_list.append(''.join(s for s in string if s != '*'))
    return string_list


def _int_to_string(value, char_map_dict=None):
    """
    将识别结果转化为string，参考：https://github.com/bai-shang/crnn_ctc_ocr.Tensorflow
    :param value:
    :param char_map_dict:
    :return:
    """
    if char_map_dict is None:
        print("error")
        #char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    assert (isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')

    for key in char_map_dict.keys():
        if char_map_dict[key] == int(value):
            return str(key)
        elif len(char_map_dict.keys()) == int(value):
            return ""
    raise ValueError('char map dict not has {:d} value. convert index to char failed.'.format(value))


def recognize_jmz(image_path, weights_path, char_dict_path, txt_file_path):
    """
    识别函数
    :param image_path: 图片所在路径
    :param weights_path: 模型保存路径
    :param char_dict_path: 字典文件存放位置
    :param txt_file_path: 包含图片名的txt文件
    :return: None
    """
    files = os.listdir(txt_file_path)
    txt_files = [txt for txt in files if txt.endswith(".txt") and txt.split(".")[0] + ".json" not in files]

    inputdata = tf.placeholder(dtype=tf.float32, shape=[1, CFG.ARCH.INPUT_SIZE[1], None, CFG.ARCH.INPUT_CHANNELS],  # 宽度可变
                               name='input')
    input_sequence_length = tf.placeholder(tf.int32, shape=[1], name='input_sequence_length')

    net = crnn_model.ShadowNet(phase='test', hidden_nums=CFG.ARCH.HIDDEN_UNITS,
                               layers_nums=CFG.ARCH.HIDDEN_LAYERS, num_classes=CFG.ARCH.NUM_CLASSES)

    inference_ret = net.inference(inputdata=inputdata, name='shadow_net', reuse=False)

    decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=inference_ret, sequence_length=input_sequence_length,  # 序列宽度可变
                                               merge_repeated=False, beam_width=1)

    # config tf saver
    saver = tf.train.Saver()

    # config tf session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    # sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    # sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH

    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
        for idx, txt_file in enumerate(txt_files):
            reg_result = {}
            txt_path = os.path.join(txt_file_path, txt_file)
            with open(txt_path, 'r') as fd:
                image_names = [line.strip() for line in fd.readlines()]
            for image_name in image_names:
                image_paths = os.path.join(image_path, image_name)
                image = cv2.imread(image_paths, cv2.IMREAD_COLOR)
                if image is None:
                    print(image_paths+'is not exist')
                    continue
                image = _resize_image(image)
                image = np.array(image, np.float32) / 127.5 - 1.0
                seq_len = np.array([image.shape[1] / 4], dtype=np.int32)
                preds = sess.run(decodes, feed_dict={inputdata: [image], input_sequence_length:seq_len})

                preds = _sparse_matrix_to_list(preds[0], char_dict_path)
                reg_result[image_name] = preds[0]
                print('Predict image {:s} result: {:s}'.format(image_name, preds[0]))
            with open(txt_path[:-4] + ".json", "w") as fw:  # 建议改为.split('.')
                json.dump(reg_result, fw)
    sess.close()

    return


if __name__ == '__main__':
    # init images
    args = init_args()
    # detect images
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"#指定在第0块GPU上跑
        # pool.apply_async(recognize, (args.image_path, args.weights_path, args.char_dict_path, os.path.join(args.txt_path,txt_file), ))
    recognize_jmz(image_path=args.image_path, weights_path=args.weights_path, 
              char_dict_path=args.char_dict_path, txt_file_path=args.txt_path)
