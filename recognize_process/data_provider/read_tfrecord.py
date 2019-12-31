#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-11-20 0:15
# @Author  : Miao Wenqiang
# @Reference    :  https://github.com/MaybeShewill-CV/CRNN_Tensorflow
#                  https://github.com/bai-shang/crnn_ctc_ocr.Tensorflow
# @File    : read_tfrecord.py
# @IDE: PyCharm
"""
Efficient tfrecords writer interface
"""


import os
import os.path as ops
import random
import glob
import tensorflow as tf
from recognize_process.config import model_config

CFG = model_config.cfg


class CrnnDataFeeder(object):

    def __init__(self, dataset_dir, char_dict_path, flags='train'):

        self._tfrecords_dir = dataset_dir
        if not ops.exists(self._tfrecords_dir):
            raise ValueError('{:s} not exist, please check again'.format(self._tfrecords_dir))

        self._dataset_flags = flags.lower()
        if self._dataset_flags not in ['train', 'test', 'val']:
            raise ValueError('flags of the data feeder should be \'train\', \'test\', \'val\'')

        self._char_dict_path = char_dict_path

    def sample_counts(self):
        tfrecords_file_paths = glob.glob('{:s}/{:s}*.tfrecords'.format(self._tfrecords_dir, self._dataset_flags))
        counts = 0

        for record in tfrecords_file_paths:
            counts += sum(1 for _ in tf.python_io.tf_record_iterator(record))

        return counts

    def _extract_features_batch(self, serialized_batch):
        features = tf.parse_example(
            serialized_batch,
            features={'images': tf.FixedLenFeature([], tf.string),
                'imagepaths': tf.FixedLenFeature([], tf.string),
                'labels': tf.VarLenFeature(tf.int64),
                 })

        bs = features['images'].shape[0]
        images = tf.decode_raw(features['images'], tf.uint8)
        w, h = tuple(CFG.ARCH.INPUT_SIZE)
        images = tf.cast(x=images, dtype=tf.float32)
        #images = tf.subtract(tf.divide(images, 128.0), 1.0)
        images = tf.reshape(images, [bs, h, -1, CFG.ARCH.INPUT_CHANNELS])

        labels = features['labels']
        labels = tf.cast(labels, tf.int32)

        imagepaths = features['imagepaths']

        return images, labels, imagepaths


    def _inputs(self, tfrecords_path, batch_size, num_threads):
        dataset = tf.data.TFRecordDataset(tfrecords_path)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        dataset = dataset.map(map_func=self._extract_features_batch, num_parallel_calls=num_threads)

        if self._dataset_flags != 'test':
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next(name='{:s}_IteratorGetNext'.format(self._dataset_flags))


    def inputs(self, batch_size):

        tfrecords_file_paths = glob.glob('{:s}/{:s}*.tfrecords'.format(\
            self._tfrecords_dir, self._dataset_flags))
        if not tfrecords_file_paths:
            raise ValueError('Dataset does not contain any tfrecords for {:s}'.format(\
                self._dataset_flags))

        random.shuffle(tfrecords_file_paths)

        return self._inputs(
            tfrecords_path=tfrecords_file_paths,
            batch_size=batch_size,
            num_threads=CFG.TRAIN.CPU_MULTI_PROCESS_NUMS
        )
