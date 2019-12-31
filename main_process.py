# -*- coding: utf-8 -*-
"""
输出结果为：
    CCFTestResultFixValidData_release.csv
"""

import argparse
import sys
import os
import time

sys.path.append('./')

from cut_twist_process import cut_twist_join  # 预处理，将身份证正反面从原始图片切分出来并旋转
from recognize_process.tools import mytest_crnn, test_crnn_jmz
from watermask_remover_and_split_data.watermask_process import WatermarkRemover
from data_correction_and_generate_csv_file.generate_test_csv_file import generate_csv


def recoginze_init_args():
    """
    初始化识别过程需要的参数
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-rc_w', '--recognize_weights_path', type=str,
                        help='Path to the pre-trained weights to use',
                        default='./recognize_process/model_save/recognize_model')
    parser.add_argument('-rc_c', '--recognize_char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored',
                        default='./recognize_process/char_map/char_map.json')
    parser.add_argument('-rc_i', '--recognize_image_path', type=str,
                        help='Path to the image to be tested',
                        default='./recognize_process/test_imgs/')
    parser.add_argument('-rc_t', '--recognize_txt_path', type=str,
                        help='Whether to display images',
                        default='./recognize_process/image_list.txt')
    parser.add_argument("--no_gen_data_chu", action="store_true", help="generate chusai new test data")
    parser.add_argument("--no_gen_data_fu", action="store_true", help="generate fusai new test data")
    parser.add_argument("--no_preprocessed", action="store_true", help="if preprocessed test data")
    parser.add_argument("--no_gan_test", action="store_true", help="test data with gan model")
    parser.add_argument("--no_gan_test_rematch", action="store_true", help="test rematch data with gan model")
    parser.add_argument("--no_rec_img", action="store_true", help="if recover img")
    parser.add_argument("--no_rec_img_rematch", action="store_true", help="if recover img")
    parser.add_argument("--no_test_data", action="store_true", help="if generate test data")
    parser.add_argument("--no_fix_img", action="store_true", help="if fix img of address and unit")
    parser.add_argument("--no_gen_txts", action="store_true", help="if txt files for recognize")
    parser.add_argument("--debug", action="store_true", help="if debug")
    parser.add_argument("--gan_chu", default="chusai_watermask_remover_model", help="model name of chusai")
    parser.add_argument("--gan_fu", default="fusai_watermask_remover_model", help="model name of fusai")
    parser.add_argument("--pool_num", default=-1, help="the number of threads for process data")
    parser.add_argument("--test_data_dir", required=True, help="the dir of test data")
    parser.add_argument("--test_experiment_name", required=True, help="the dir of test data")
    parser.add_argument("--gan_ids", required=True, help="-1 for cpu, 0 or 0,1.. for GPU")

    return parser.parse_args()


if __name__ == '__main__':
    args = recoginze_init_args()
    origin_img_path = args.test_data_dir
    time_log = time.strftime("%y_%m_%d_%H_%M_%S")
    header_dir = os.path.join("./data_temp", args.test_experiment_name + "_" + time_log)
    if not os.path.exists(header_dir):
        os.makedirs(header_dir)
    cut_twisted_save_path = os.path.join(header_dir, 'data_cut_twist')  # 切分、旋转后数据保存路径
    cut_twist_template_names = ['./cut_twist_process/template/fan_blurred_fan.jpg',  # 0 反面反
                                './cut_twist_process/template/fan_blurred_zheng.jpg',  # 1 反面正
                                './cut_twist_process/template/zheng_blurred_fan.jpg',  # 2 正面反
                                './cut_twist_process/template/zheng_blurred_zheng.jpg',  # 3 正面正
                                './cut_twist_process/template/zheng_new.jpg',  # 4 新水印正面
                                './cut_twist_process/template/fan_new.jpg'  # 5 新水印反面
                                ]  # 模板图片路径
    # 切分身份证
    cut_twist_join.process_cut_twist_imgs(img_path=origin_img_path, template_names=cut_twist_template_names,
                                          save_path=cut_twisted_save_path, norm_parm=[0.95, 0.95, 0.7, 0.7])
    # 去水印和对图片进行切割和处理
    watermask_handler = WatermarkRemover(args, header_dir, cut_twisted_save_path)
    watermask_handler.watermask_remover_run()
    recognize_image_path = os.path.join(header_dir, "test_data_preprocessed")
    recognize_txt_path = os.path.join(header_dir, "test_data_txts")
    test_crnn_jmz.recognize_jmz(image_path=recognize_image_path, weights_path=args.recognize_weights_path,
                                char_dict_path=args.recognize_char_dict_path, txt_file_path=recognize_txt_path)
    origin_watermask_removed_img_path = os.path.join(header_dir, "recover_image_fu_dir")
    generate_csv(origin_watermask_removed_img_path, recognize_txt_path, "./")
