#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-11-20 0:12
# @Author  : Miao Wenqiang
# @Reference    :  https://github.com/MaybeShewill-CV/CRNN_Tensorflow
#                  https://github.com/bai-shang/crnn_ctc_ocr.Tensorflow
#                  https://github.com/balancap/SSD-Tensorflow
# @File    : train_crnn.py
# @IDE: PyCharm Community Edition

import sys
sys.path.append('./')
print(sys.path)

import os
import os.path as ops
import time
import argparse

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops

from recognize_process.crnn_model import crnn_model
from recognize_process.config import model_config
from recognize_process.data_provider import read_tfrecord

CFG = model_config.cfg


def init_args():
    """
    参数初始化
    :return: None
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_dir', type=str,
                        help='Directory containing train_features.tfrecords')
    parser.add_argument('-w', '--weights_path', type=str,
                        help='Path to pre-trained weights to continue training')
    parser.add_argument('-s', '--save_path', type=str,
                        help='Path to logs and ckpt models')
    parser.add_argument('-c', '--char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored',
                        default='./recognize_process/char_map/char_map.json')

    return parser.parse_args()


##########################################################################
def apply_with_random_selector(x, func, num_cases):
    """
    随机选择数据增强方式，参考：
    https://github.com/balancap/SSD-Tensorflow
    :param x:
    :param func:
    :param num_cases:
    :return:
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
            for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, scope=None):
    """
    随机进行图像增强（亮度、对比度操作）
    :param image: 输入图片
    :param color_ordering:模式
    :param scope: 命名空间
    :return: 增强后的图片
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if color_ordering == 0:  # 模式0.先调整亮度，再调整对比度
            rand_temp = random_ops.random_uniform([], -55, 20, seed=None) # [-70, 30] for generate img, [-50, 20] for true img 
            image = math_ops.add(image, math_ops.cast(rand_temp, dtypes.float32))
            image = tf.image.random_contrast(image, lower=0.45, upper=1.5) # [0.3, 1.75] for generate img, [0.45, 1.5] for true img 
        else:
            image = tf.image.random_contrast(image, lower=0.45, upper=1.5)
            rand_temp = random_ops.random_uniform([], -55, 30, seed=None)
            image = math_ops.add(image, math_ops.cast(rand_temp, dtypes.float32))

        # The random_* ops do not necessarily clamp.
        print(color_ordering)
        return tf.clip_by_value(image, 0.0, 255.0)  # 限定在0-255
##########################################################################




def train_shadownet(dataset_dir, weights_path, char_dict_path, save_path):
    """
    训练网络，参考：
    https://github.com/MaybeShewill-CV/CRNN_Tensorflow
    :param dataset_dir: tfrecord文件路径
    :param weights_path: 要加载的预训练模型路径
    :param char_dict_path: 字典文件路径
    :param save_path: 模型保存路径
    :return: None
    """
    # prepare dataset
    train_dataset = read_tfrecord.CrnnDataFeeder(
        dataset_dir=dataset_dir, char_dict_path=char_dict_path, flags='train')

    train_images, train_labels, train_images_paths = train_dataset.inputs(
        batch_size=CFG.TRAIN.BATCH_SIZE)
####################添加数据增强##############################
    # train_images = tf.multiply(tf.add(train_images, 1.0), 128.0)   # removed since read_tfrecord.py is changed
    tf.summary.image('original_image', train_images)   # 保存到log，方便测试观察
    images = apply_with_random_selector(
        train_images,
        lambda x, ordering: distort_color(x, ordering),
        num_cases=2)  #
    images = tf.subtract(tf.divide(images, 127.5), 1.0)  # 转化到【-1，1】 changed 128.0 to 127.5 
    train_images = tf.clip_by_value(images, -1.0, 1.0)
    tf.summary.image('distord_turned_image', train_images)
################################################################


    # declare crnn net
    shadownet = crnn_model.ShadowNet(phase='train',hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS, num_classes=CFG.ARCH.NUM_CLASSES)
    # set up training graph
    with tf.device('/gpu:0'):
        # compute loss and seq distance
        train_inference_ret, train_ctc_loss = shadownet.compute_loss(inputdata=train_images,
            labels=train_labels, name='shadow_net', reuse=False)

        # set learning rate
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=CFG.TRAIN.LEARNING_RATE,
            global_step=global_step, decay_steps=CFG.TRAIN.LR_DECAY_STEPS,
            decay_rate=CFG.TRAIN.LR_DECAY_RATE, staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
            #    momentum=0.9).minimize(loss=train_ctc_loss, global_step=global_step)
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=\
                learning_rate).minimize(loss=train_ctc_loss, global_step=global_step)
            # 源代码优化器是momentum，改成adadelta，与CRNN论文一致


    # Set tf summary
    os.makedirs(save_path, exist_ok=True)
    tf.summary.scalar(name='train_ctc_loss', tensor=train_ctc_loss)
    tf.summary.scalar(name='learning_rate', tensor=learning_rate)
    merge_summary_op = tf.summary.merge_all()

    # Set saver configuration
    saver = tf.train.Saver()
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(save_path, model_name)

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(save_path)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    with sess.as_default():
        epoch = 0
        if weights_path is None:
            print('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            print('Restore model from {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)
            epoch = sess.run(tf.train.get_global_step())

        cost_history = [np.inf]
        while epoch < train_epochs:
            epoch += 1
            _, train_ctc_loss_value, merge_summary_value, learning_rate_value = sess.run(
                [optimizer, train_ctc_loss, merge_summary_op, learning_rate])

            if (epoch+1) % CFG.TRAIN.DISPLAY_STEP == 0:
                print('Leearning_rate = {:9f}         Step_Train: {:d}         cost= {:9f}'.format(\
                    learning_rate_value, epoch+1, train_ctc_loss_value))
                # record history train ctc loss
                cost_history.append(train_ctc_loss_value)
                 # add training sumary
                summary_writer.add_summary(summary=merge_summary_value, global_step=epoch)

            if (epoch+1) % CFG.TRAIN.SAVE_STEPS == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

    return np.array(cost_history[1:])  # Don't return the first np.inf


if __name__ == '__main__':
    # init args
    args = init_args()
    # time.sleep(7200)
    # time.sleep(4800)
    print('start')
    train_shadownet(
        dataset_dir=args.dataset_dir,
        weights_path=args.weights_path,
        char_dict_path=args.char_dict_path,
        save_path=args.save_path
    )
