#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-11-20 0:25
# @Author  : Miao Wenqiang
# @Reference    :  https://github.com/MaybeShewill-CV/CRNN_Tensorflow
#                  https://github.com/bai-shang/crnn_ctc_ocr.Tensorflow
# @File    : write_tfrecord.py
# @IDE: PyCharm
"""
Efficient tfrecords writer interface
"""


import os
import sys
import os.path as ops
import tensorflow as tf
import argparse
from multiprocessing import Manager
from multiprocessing import Process
import time
import tqdm
import json
import cv2

from recognize_process.config import model_config

CFG = model_config.cfg

_SAMPLE_INFO_QUEUE = Manager().Queue()
_SENTINEL = ("", [])


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, help='The origin synth90k dataset_dir', default=None)
    parser.add_argument('-s', '--save_dir', type=str, help='The generated tfrecords save dir', default=None)
    parser.add_argument('-c', '--char_dict_path', type=str, default='./recognize_process/char_map/char_map.json',
                        help='The char dict file path. If it is None the char dict will be'
                             'generated automatically in folder data/char_dict')
    parser.add_argument('-a', '--anno_file_path', type=str, default=None,
                        help='The ord map dict file path. If it is None the ord map dict will be'
                             'generated automatically in folder data/char_dict')

    return parser.parse_args()



def _string_to_int(char_map_path, label):
    # convert string label to int list by char map
    char_map_dict = json.load(open(char_map_path, 'r'))
    int_list = []
    for c in label:
        #if c not in char_map_dict:
            #print(c, 'is not in char_map')
            #continue
        int_list.append(char_map_dict[c])
    return int_list


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_int = True
    for val in value:
        if not isinstance(val, int):
            is_int = False
            value_tmp.append(int(float(val)))
    if not is_int:
        value = value_tmp
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, bytes):
        if not isinstance(value, list):
            value = value.encode('utf-8')
        else:
            value = [val.encode('utf-8') for val in value]
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))



def _is_valid_jpg_file(image_path):
    if not ops.exists(image_path):
        return False

    file = open(image_path, 'rb')
    data = file.read(11)
    if data[:4] != '\xff\xd8\xff\xe0' and data[:4] != '\xff\xd8\xff\xe1':
        file.close()
        return False
    if data[6:] != 'JFIF\0' and data[6:] != 'Exif\0':
        file.close()
        return False
    file.close()

    file = open(image_path, 'rb')
    file.seek(-2, 2)
    if file.read() != '\xff\xd9':
        file.close()
        return False

    file.close()

    return True


def _resize_image(img):
    dst_width = CFG.ARCH.INPUT_SIZE[0]
    dst_height = CFG.ARCH.INPUT_SIZE[1]
    h_old, w_old, _ = img.shape
    height = dst_height
    width = int(w_old * height / h_old)
    if width < dst_width:
        left_padding = int((dst_width - width)/2)
        right_padding = dst_width - width - left_padding
        resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        resized_img = cv2.copyMakeBorder(resized_img, 0, 0, left_padding, right_padding,
            cv2.BORDER_CONSTANT, value=[255, 255, 255])
    else:
        resized_img = cv2.resize(img, (dst_width, height), interpolation=cv2.INTER_CUBIC)

    return resized_img


def _init_data_queue(img_dir, anno_file_path, char_map_path, writer_process_nums, dataset_flag='train'):
    print('Start filling {:s} dataset sample information queue...'.format(dataset_flag))
    t_start = time.time() #开始处理，计时

    annotation_infos = []

    num_lines = sum(1 for _ in open(anno_file_path, 'r')) # 图片及标注行数
    with open(anno_file_path, 'r', encoding='utf-8') as file:
        for line in tqdm.tqdm(file, total=num_lines):
            image_name, label_index = line.rstrip('\r').rstrip('\n').split(' ')
            #img_dir = img_dir.strip()
            #image_name = image_name.upper()
            image_path = ops.join(img_dir, image_name.strip()) # 图片地址
            label_index = label_index.lower()
            label_index = _string_to_int(char_map_path, label_index) # label本是字符串，转为int
            if not ops.exists(image_path):
                print('Example image {:s} not exist'.format(image_path))
                continue
                #raise ValueError('Example image {:s} not exist'.format(image_path))
            annotation_infos.append((image_path, label_index)) # 结果

    for annotation_info in tqdm.tqdm(annotation_infos):
        image_path = annotation_info[0]
        label_index = annotation_info[1]
        try:
            _SAMPLE_INFO_QUEUE.put((image_path, label_index))
        except IndexError:
            print('Lexicon doesn\'t contain lexicon index {:d}'.format(label_index))
            continue
    for i in range(writer_process_nums): # 添加结束标志
        _SAMPLE_INFO_QUEUE.put(_SENTINEL)
    print('Complete filling dataset sample information queue[current size: {:d}], cost time: {:.5f}s'.format(
        _SAMPLE_INFO_QUEUE.qsize(), time.time() - t_start))


def _write_tfrecords(tfrecords_writer):
    while True:
        sample_info = _SAMPLE_INFO_QUEUE.get()
        if sample_info == _SENTINEL:
            print('Process {:d} finished writing work'.format(os.getpid()))
            tfrecords_writer.close()
            break

        sample_path = sample_info[0]
        sample_label = sample_info[1]

        if _is_valid_jpg_file(sample_path):
            print('Image file: {:d} is not a valid jpg file'.format(sample_path))
            continue

        try:
            image = cv2.imread(sample_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = _resize_image(image)
            image = image.tostring()
        except IOError as err:
            print(err)
            continue
        sample_path = sample_path if sys.version_info[0] < 3 else sample_path.encode('utf-8')

        features = tf.train.Features(feature={
            'labels': _int64_feature(sample_label),
            'images': _bytes_feature(image),
            'imagepaths': _bytes_feature(sample_path)
        })
        tf_example = tf.train.Example(features=features)
        tfrecords_writer.write(tf_example.SerializeToString())
        #print('Process: {:d} get sample from sample_info_queue[current_size={:d}], '
        #          'and write it to local file at time: {}'.format(
        #           os.getpid(), _SAMPLE_INFO_QUEUE.qsize(), time.strftime('%H:%M:%S')))




class CrnnDataProducer(object):

    def __init__(self, dataset_dir, char_dict_path=None, anno_file_path=None,
                 writer_process_nums=4, dataset_flag='train'):
        if not ops.exists(dataset_dir):
            raise ValueError('Dataset dir {:s} not exist'.format(dataset_dir))

        # Check image source data
        self._dataset_dir = dataset_dir
        self._annotation_file_path = anno_file_path
        self._char_dict_path = char_dict_path
        self._writer_process_nums = writer_process_nums
        self._dataset_flag = dataset_flag

    def generate_tfrecords(self, save_dir):
        # make save dirs
        os.makedirs(save_dir, exist_ok=True)
        # generate training example tfrecords
        print('Generating training sample tfrecords...')
        t_start = time.time()
        print('Start write tensorflow records for {:s}...'.format(self._dataset_flag))

        process_pool = []
        tfwriters = []
        for i in range(self._writer_process_nums):
            tfrecords_save_name = '{:s}_{:d}.tfrecords'.format(self._dataset_flag, i + 1)
            tfrecords_save_path = ops.join(save_dir, tfrecords_save_name)

            tfrecords_io_writer = tf.python_io.TFRecordWriter(path=tfrecords_save_path)
            process = Process(target=_write_tfrecords, name='Subprocess_{:d}'.format(i + 1),
                args=(tfrecords_io_writer,))
            process_pool.append(process)
            tfwriters.append(tfrecords_io_writer)
            process.start()

        for process in process_pool:
            process.join()
        print('Generate {:s} sample tfrecords complete, cost time: {:.5f}'\
              .format(self._dataset_flag, time.time() - t_start))
        return




def write_tfrecords(dataset_dir, char_dict_path, anno_file_path, save_dir, writer_process_nums, dataset_flag):
    assert ops.exists(dataset_dir), '{:s} not exist'.format(dataset_dir)
    os.makedirs(save_dir, exist_ok=True)
    # test crnn data producer
    _init_data_queue(img_dir=dataset_dir, anno_file_path=anno_file_path,char_map_path=\
        char_dict_path, writer_process_nums=writer_process_nums, dataset_flag=dataset_flag)

    producer = CrnnDataProducer(dataset_dir=dataset_dir, char_dict_path=char_dict_path,
        anno_file_path=anno_file_path, writer_process_nums=writer_process_nums, dataset_flag=dataset_flag)

    producer.generate_tfrecords(save_dir=save_dir)


if __name__ == '__main__':
    args = init_args()

    write_tfrecords(dataset_dir=args.dataset_dir, char_dict_path=args.char_dict_path,
        anno_file_path=args.anno_file_path, save_dir=args.save_dir,
        writer_process_nums = 10, dataset_flag='train')


