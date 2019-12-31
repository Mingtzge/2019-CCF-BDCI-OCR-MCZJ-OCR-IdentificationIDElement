import os
from multiprocessing import Pool


def run(idx, test_img_dst_path, test_img_names, img_names):
    fw = open(os.path.join(test_img_dst_path, "images_list" + "_" + str(idx) + ".txt"), "w")
    img_name = img_names[idx][:-5]
    for te_img in test_img_names:
        if img_name in te_img:
            fw.write(te_img + "\n")
    fw.close()


def generate_txts(origin_img_path, test_img_path, test_img_dst_path, pool_num):
    """
    :param origin_img_path: 原始的图片的路径
    :param test_img_path: 用于识别的图片的路径
    :param test_img_dst_path: 生成的txts图片列表存放路径
    :param pool_num: 进程数
    描述:将每套身份证的10个元素图片名,保存在同一个txt文件中,用于文字识别
    """
    if not os.path.exists(test_img_dst_path):
        os.makedirs(test_img_dst_path)
    imgs = os.listdir(origin_img_path)
    test_img_names = os.listdir(test_img_path)
    img_names = [im.split("_")[0] for im in imgs if im.split("_")[1][0] == "0"] # 提取图片名
    epoch_count = len(img_names)
    if pool_num > 0:
        pool = Pool(pool_num)
        for idx in range(epoch_count):
            pool.apply_async(run, (idx, test_img_dst_path, test_img_names, img_names,))
        pool.close()
        pool.join()
    else:
        for idx in range(epoch_count):
            run(idx, test_img_dst_path, test_img_names, img_names)
    print("txts generation finished")
