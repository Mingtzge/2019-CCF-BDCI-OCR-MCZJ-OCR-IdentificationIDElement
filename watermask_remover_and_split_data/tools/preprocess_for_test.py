import cv2
import numpy as np
import os
from multiprocessing import Pool


def _resize_image(img, dst_height):
    h_old = img.shape[0]
    w_old = img.shape[1]
    height = dst_height
    width = int(w_old * height / h_old)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    return resized_img


def preprocess_one_img(img):
    resize_img = _resize_image(img, 32)  # 修改图片的高度
    # 对图片进行滤波处理
    resize_img = cv2.normalize(resize_img, dst=None, alpha=230, beta=20, norm_type=cv2.NORM_MINMAX)
    resize_img = cv2.bilateralFilter(src=resize_img, d=3, sigmaColor=200, sigmaSpace=10)
    resize_img = cv2.filter2D(resize_img, -1, kernel=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32))
    return resize_img


def cv_imread(image_path):
    cv_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    return cv_img


def cv_imwrite(write_path, img):
    cv2.imencode('.jpg', img, )[1].tofile(write_path)
    return


def preprocess_imgs(img_path, save_path, pool_num):
    """
    :param img_path: 处理的图片路径
    :param save_path: 保存路径
    :param pool_num: 处理进程数
    描述:主要是对图片进行滤波处理和尺寸变换(模型对输入图片的尺寸有求),提高识别的准确率
    """
    img_names = os.listdir(img_path)
    if not os.path.exists(img_path):
        print("not exists ", img_path, " exit...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    params = []
    for img_name in img_names:
        params.append((img_path, save_path, img_name))
    if pool_num > 0:
        pool = Pool(pool_num)
        pool.map(pre_run, params)
        pool.close()
        pool.join()
    else:
        for param in params:
            pre_run(param)


def pre_run(params):
    run(params[0], params[1], params[2])


def run(img_path, save_path, img_name):
    img = cv_imread(os.path.join(img_path, img_name))
    img_blurred = preprocess_one_img(img)
    cv_imwrite(os.path.join(save_path, img_name), img_blurred)
