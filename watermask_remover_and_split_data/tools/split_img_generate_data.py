import os
from multiprocessing import Pool
import cv2

# 身份证上面各个元素的准确坐标,长宽,序号,用于从图片企鹅个
issuing_unit = {
    "x_d": 167,
    "y_d": 191,
    "w": 230,
    "h": 40,
    "index": 9
}
effective_data = {
    "x_d": 167,
    "y_d": 227,
    "w": 192,
    "h": 19,
    "index": 10
}
name = {
    "x_d": 85,
    "y_d": 39,
    "w": 106,
    "h": 24,
    "index": 1
}
gender = {
    "x_d": 87,
    "y_d": 72,
    "w": 24,
    "h": 24,
    "index": 3
}
nationality = {
    "x_d": 185,
    "y_d": 72,
    "w": 121,
    "h": 25,
    "index": 2
}
birthday_year = {
    "x_d": 84,
    "y_d": 105,
    "w": 47,
    "h": 21,
    "index": 4
}
birthday_month = {
    "x_d": 147,
    "y_d": 105,
    "w": 31,
    "h": 23,
    "index": 5
}
birthday_day = {
    "x_d": 198,
    "y_d": 105,
    "w": 29,
    "h": 22,
    "index": 6
}
address = {
    "x_d": 82,
    "y_d": 138,
    "w": 210,
    "h": 64,
    "index": 7
}
id_card = {
    "x_d": 131,
    "y_d": 221,
    "w": 246,
    "h": 24,
    "index": 8
}


def match_img(image, template, value):
    """
    :param image: 图片
    :param template: 模板
    :param value: 阈值
    :return: 水印坐标
    描述:用于获得这幅图片模板对应的位置坐标,用途:校准元素位置信息
    """
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = value
    min_v, max_v, min_pt, max_pt = cv2.minMaxLoc(res)
    if max_v < threshold:
        return False
    if not max_pt[0] in range(10, 40) or max_pt[1] > 20:
        return False
    return max_pt


def crop_img(mark_point, args, ori_img, save_path, seq, label, type_c):
    """
    :param mark_point: 信标点
    :param args: 元素相关参数,坐标,长宽
    :param ori_img: 图片
    :param save_path: 切割之后的元素保存路径
    :param seq: 序号
    :param label: 标记
    :param type_c: 类型(没有用到)
    :return:
    """
    try:
        x_p = mark_point[0] + args["x_d"]
        y_p = mark_point[1] + args["y_d"]
        c_img = ori_img[y_p:y_p + args["h"], x_p: x_p + args["w"]]
        c_img_save_path = os.path.join(save_path, "%s_%s_%s.jpg" % (str(seq), label, str(args["index"])))
        cv2.imwrite(c_img_save_path, c_img)
    except():
        print("crop except")
        return


def generate_data(ori_img_path, template, save_path, flag, thr_value, seq, label, type_c):
    """
    :param ori_img_path: 图片路径
    :param template: 模板
    :param save_path: 保存路径
    :param flag: 正反面标记
    :param thr_value: 匹配阈值
    :param seq: 序号
    :param label: 标记
    :param type_c: 类型
    :return: 无
    """
    ori_img = cv2.imread(ori_img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
    mark_point = match_img(ori_img, template, thr_value) # 获取各个元素的参考坐标
    if mark_point is False:
        print(" failed")
        return
    mark_point = (max(0, mark_point[0] - 20), mark_point[1])
    if flag == "0":
        # 截取背面两种元素
        crop_img(mark_point, issuing_unit, ori_img, save_path, seq, label, type_c)
        crop_img(mark_point, effective_data, ori_img, save_path, seq, label, type_c)
    else:
        # 截取正面两种元素
        crop_img(mark_point, name, ori_img, save_path, seq, label, type_c)
        crop_img(mark_point, gender, ori_img, save_path, seq, label, type_c)
        crop_img(mark_point, birthday_year, ori_img, save_path, seq, label, type_c)
        crop_img(mark_point, birthday_month, ori_img, save_path, seq, label, type_c)
        crop_img(mark_point, birthday_day, ori_img, save_path, seq, label, type_c)
        crop_img(mark_point, address, ori_img, save_path, seq, label, type_c)
        crop_img(mark_point, id_card, ori_img, save_path, seq, label, type_c)
        crop_img(mark_point, nationality, ori_img, save_path, seq, label, type_c)


def run_gen_test_data(final_save_path, template_base_path, origin_img_path, pool_num):
    """
    :param final_save_path: 保存路径
    :param template_base_path: 模板图片路径
    :param origin_img_path: 待切割图片的路径
    :param pool_num: 进程数量
    :return: 无
    """
    template_img = cv2.imread(template_base_path, 0)
    img_names = os.listdir(origin_img_path)
    if not os.path.exists(final_save_path):
        os.makedirs(final_save_path)
    pool = Pool(1)
    if pool_num > 0:
        pool = Pool(pool_num)
    for count, img_name in enumerate(img_names):
        img_path = os.path.join(origin_img_path, img_name)
        names = img_name.split("_")
        if pool_num > 0:
            pool.apply_async(generate_data,(img_path, template_img, final_save_path, names[1][0], 0.2, count, img_name[:-4], "Test", ))
        else:
            generate_data(img_path, template_img, final_save_path, names[1][0], 0.2, count, img_name[:-4], "Test")
    pool.close()
    pool.join()
