import cv2
import numpy as np
from multiprocessing import Pool
from watermask_remover_and_split_data.tools.split_img_generate_data import run_gen_test_data
from watermask_remover_and_split_data.tools.fix_img_address_unit import fix_address_unit
from watermask_remover_and_split_data.tools.preprocess_for_test import preprocess_imgs
from watermask_remover_and_split_data.tools.extract_test_img_to_txts import generate_txts
import os

size_chu = (512, 512)
size_fu = (256, 256)


def merge_img(image1, image2):
    """
    描述:将两张图片水平拼接,去水印时要求的格式
    """
    h1, w1 = image1.shape
    h2, w2 = image2.shape
    if h1 != h2 or w1 != w2:
        image2 = cv2.resize(image2, (w1, h1))
    image3 = np.hstack([image2, image1])
    return image3


class WatermarkRemover:
    """
    去水印的类,主要包含功能:
    1,从图片中提取水印部分,去除水印并复原
    2,对图片进行切割,提取身份证中各个元素部分
    3,将签发机关和地址转化成一行
    4,对图像进行识别前预处理
    """

    def __init__(self, args, header_dir, ori_img_path):
        """
        :param args: 配置参数
        :param header_dir: 数据文件存放目录
        :param ori_img_path: 原始数据(切割和旋转之后)的路径
        """
        self.header_dir = header_dir
        self.ori_img_path = ori_img_path
        self.args = args
        self.rematch = False
        self.pool_num = int(args.pool_num)
        self.gan_ids = args.gan_ids
        self.pixel_mode = args.gan_chu
        dirs = ["chusai_data_for_watermark_remove/test", "fuusai_data_for_watermark_remove/test", "gan_result_chu_dir",
                "gan_result_fu_dir", "recover_image_chu_dir", "recover_image_fu_dir", "train_data_dir",
                "test_data_preprocessed", "test_data_txts"]
        self.test_data_dst_path = os.path.join(header_dir, "chusai_data_for_watermark_remove")
        self.gan_result_dir = os.path.join(header_dir, "gan_result_chu_dir")
        self.recover_image_dir = os.path.join(header_dir, "recover_image_chu_dir")
        self.train_data_dir = os.path.join(header_dir, "train_data_dir")
        self.fix_bak_data_dir = os.path.join(header_dir, "fix_bak_data")
        self.preprocessed_dir = os.path.join(header_dir, "test_data_preprocessed")
        self.test_data_txts_dir = os.path.join(header_dir, "test_data_txts")
        self.roi_img_path = './watermask_remover_and_split_data/template_imgs/chusai_watermask_template.jpg'
        self.roi_rematch_img_path = "./watermask_remover_and_split_data/template_imgs/fusai_watermask_template.jpg"
        self.base_template = "./watermask_remover_and_split_data/template_imgs/origin_img_location_marker_template.jpg"
        for sub_dir in dirs:
            if not os.path.exists(os.path.join(header_dir, sub_dir)):
                os.makedirs(os.path.join(header_dir, sub_dir))
        ret, roi_img_bin = cv2.threshold(cv2.cvtColor(cv2.imread(self.roi_img_path), cv2.COLOR_RGB2GRAY), 175, 255,
                                         cv2.THRESH_BINARY)
        self.width, self.height = roi_img_bin.shape[::-1]

    def match_img(self, image, target, value, rematch=False):
        """
        :param image: 原始图片
        :param target: 匹配模板
        :param value: 匹配阈值
        :param rematch: false,初赛水印,true复赛水印
        :return: 水印外轮廓坐标,原始图片灰度图,水印内轮廓
        """
        img_rgb = cv2.imread(image)
        h, w, c = img_rgb.shape
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(target, 0)
        th, tw = template.shape
        max_v1 = 0
        if not rematch:
            template = template[16:56, 20:186]
        else:
            template = template[18:107, 19:106]
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = value
        min_v, max_v, min_pt, max_pt = cv2.minMaxLoc(res)
        if max_v < threshold:
            return False, False, False
        if not rematch:
            template1 = cv2.imread(self.roi_rematch_img_path, 0)
            template1 = template1[18:107, 19:106]
            res1 = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            min_v1, max_v1, min_pt1, max_pt1 = cv2.minMaxLoc(res1)
            if max_v < max_v1:  # 避免两种水印匹配重叠的情况
                return False, False, False
        if not rematch:
            x = 20
            y = 16
        else:
            x = 19
            y = 18
        ori_pt = (min(w - tw - 1, max(max_pt[0] - x, 0)), max(0, min(max_pt[1] - y, h - th - 1)))
        return ori_pt, img_gray, max_pt

    def gen_align_real_test_data(self):
        """
        描述:截取图片水印部分,用于gan网络去水印,多进程实现
        """
        if self.pool_num > 0:
            origin_img_names = os.listdir(self.ori_img_path)
            pool = Pool(self.pool_num)
            if self.args.debug:
                pool.map(self.gen_test_data_run, origin_img_names[:self.pool_num])
            else:
                pool.map(self.gen_test_data_run, origin_img_names)
            pool.close()
            pool.join()
        else:
            for img in os.listdir(self.ori_img_path):
                self.gen_test_data_run(img)

    def gen_test_data_run(self, ori_name):
        """
        :param ori_name: 处理的图片名
        描述: 截取图片水印部分执行函数, 并记录水印位置,用于后期纠正使用
        """
        img_path = os.path.join(self.ori_img_path, ori_name)
        pt, img_gray, m_p = self.match_img(img_path, self.roi_img_path, 0.3, self.rematch)
        if pt is False:
            # print(img_path)
            # continue
            return
        new_img = merge_img(img_gray[pt[1]:pt[1] + self.height, pt[0]:pt[0] + self.width],
                            img_gray[pt[1]:pt[1] + self.height, pt[0]:pt[0] + self.width])
        if not self.rematch:
            flag = "_chu_" + str(pt[0]) + "_" + str(pt[1]) + "_" + str(m_p[0]) + "_" + str(m_p[1]) + "_chu_"
        else:
            flag = "_fu_" + str(pt[0]) + "_" + str(pt[1]) + "_" + str(m_p[0]) + "_" + str(m_p[1]) + "_fu_"
        cv2.imwrite(os.path.join(self.test_data_dst_path, "test", ori_name[:-4] + flag + ".jpg"), new_img)

    def gan_gen_result(self):
        """
        描述:执行去水印操作
        复印无效  禁止复印 依次去除
        """
        if not self.rematch:
            size_ = size_chu
        else:
            size_ = size_fu
        test_img_count = len(os.listdir(os.path.join(self.test_data_dst_path, "test")))
        if test_img_count > 0 and os.path.exists(self.test_data_dst_path):
            os.system(
                "python ./pytorch-CycleGAN-and-pix2pix/test.py --model pix2pix --direction AtoB --dataroot %s --name %s\
                  --num_test %s --results_dir %s --load_size %s --crop_size %s --checkpoints_dir %s --gpu_ids %s" % (
                    self.test_data_dst_path, self.pixel_mode, test_img_count, self.gan_result_dir, size_[0], size_[1],
                    "./pytorch-CycleGAN-and-pix2pix/models_data", self.gan_ids))
        else:
            print("there are something wrong in test_data_dst_path, exit...")
            exit(0)

    def recover_origin_img(self):
        """
        描述: 将去完水印的图恢复到原图上
        """
        result_dir = os.path.join(self.gan_result_dir, self.pixel_mode, "test_latest", "images")
        if not os.path.exists(result_dir):
            print("not exists gan result dir, exit...")
            exit(0)
        result_img_names = os.listdir(result_dir)
        recovered_imgs = []
        for result_img_name in result_img_names:
            if "fake" in result_img_name:
                target_img_name = result_img_name.split("_")[0] + "_" + result_img_name.split("_")[1] + ".jpg"
                if self.rematch:
                    target_img_name = result_img_name.split("_fu_")[0] + ".jpg"
                if not self.rematch:
                    if not "_chu_" in result_img_name:
                        print("img name invalid!!-----------------------------<>----------------------------",
                              result_img_name)
                        continue
                    pts = result_img_name.split("_chu_")[1].split("_")
                else:
                    if not "_fu_" in result_img_name:
                        print("img name invalid!!-----------------------------<>----------------------------",
                              result_img_name)
                        continue
                    pts = result_img_name.split("_fu_")[1].split("_")
                pt = (int(pts[0]), int(pts[1]))
                result_img = cv2.resize(
                    cv2.cvtColor(cv2.imread(os.path.join(result_dir, result_img_name)), cv2.COLOR_BGR2GRAY),
                    (self.width, self.height))
                target_img = cv2.cvtColor(cv2.imread(os.path.join(self.ori_img_path, target_img_name)),
                                          cv2.COLOR_BGR2GRAY)
                # target_img[pt[1]:height, pt[0]:width] = result_img #图片替换
                try:
                    for i in range(self.height):
                        for j in range(self.width):
                            target_img[pt[1] + i, pt[0] + j] = result_img[i, j]
                            # for p in water_points:
                    #    target_img[p[0] + pt[1], p[1] + pt[0]] = result_img[p[0], p[1]]
                    cv2.imwrite(os.path.join(self.recover_image_dir, result_img_name[:-11] + ".jpg"), target_img)
                    recovered_imgs.append(target_img_name)
                except Exception:
                    print("恢复图片异常：", result_img_name)
        ori_imgs = os.listdir(self.ori_img_path)
        print("原始图片数量:", len(ori_imgs), "恢复图片数量:", len(recovered_imgs))
        if len(ori_imgs) > len(recovered_imgs) and not self.args.debug:
            for img in ori_imgs:
                if img not in recovered_imgs:
                    os.system("cp %s %s" % (os.path.join(self.ori_img_path, img), self.recover_image_dir + "/"))

    def watermask_remover_run(self):
        args = self.args
        if not args.no_gen_data_chu:  # 生成用于去水印(复印无效)的测试集
            print("running gen_data ....")
            self.gen_align_real_test_data()
        if not args.no_gan_test:  # 启动训练好的gan模型去水印
            print("running gan_test ....")
            self.gan_gen_result()
        if not args.no_rec_img:  # 将去好水印的图片恢复到原图
            print("running rec_img ....")
            self.recover_origin_img()
            self.ori_img_path = self.recover_image_dir  # 将原始图片路径改为去过水印之后的图片路径
        if not args.no_gen_data_fu:  # 生成用于去水印(禁止复印)的测试集
            self.rematch = True
            print("running gen_data_fu ....")
            self.roi_img_path = self.roi_rematch_img_path
            ret, roi_img_bin = cv2.threshold(cv2.cvtColor(cv2.imread(self.roi_img_path), cv2.COLOR_RGB2GRAY), 175, 255,
                                             cv2.THRESH_BINARY)
            self.width, self.height = roi_img_bin.shape[::-1]  # 记录模板尺寸
            self.test_data_dst_path = os.path.join(self.header_dir, "fuusai_data_for_watermark_remove")
            self.gen_align_real_test_data()
        if not args.no_gan_test_rematch:  # 启动训练好的gan模型去水印
            print("running gan_test_fu ....")
            self.pixel_mode = args.gan_fu
            self.gan_result_dir = os.path.join(self.header_dir, "gan_result_fu_dir")
            self.gan_gen_result()
        if not args.no_rec_img_rematch:  # 将去好水印的图片恢复到原图,只是像素点还原
            print("running rec_img_fu ....")
            self.recover_image_dir = os.path.join(self.header_dir, "recover_image_fu_dir")
            self.recover_origin_img()
        if not args.no_test_data:  # 生成用于文字识别测试的数据
            print("running test_data ....")
            run_gen_test_data(self.train_data_dir, self.base_template, self.recover_image_dir, self.pool_num)
        if not args.no_fix_img:  # 将地址和签发机关的数据拆分成一行
            print("running fix_data ....")
            fix_address_unit(self.train_data_dir, self.fix_bak_data_dir, self.pool_num)
        if not args.no_preprocessed:  # 图像预处理
            print("running preprocessed_data ....")
            preprocess_imgs(self.train_data_dir, self.preprocessed_dir, self.pool_num)
        if not args.no_gen_txts:  # 生成图片的列表文件,用于识别
            print("running generate_txts ....")
            generate_txts(self.ori_img_path, self.preprocessed_dir, self.test_data_txts_dir, self.pool_num)
