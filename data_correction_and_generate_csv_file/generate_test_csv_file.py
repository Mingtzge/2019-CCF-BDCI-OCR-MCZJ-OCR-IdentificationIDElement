import json
import csv
import os
import cv2
from data_correction_and_generate_csv_file.currect_tools.valid_data import ValidData
from data_correction_and_generate_csv_file.currect_tools.birthday_id_number import BirIDNumFix
from data_correction_and_generate_csv_file.currect_tools.address_correct import AddressCorrect

# 56个民族的数据, 用于纠正识别结果
nationality = "汉族 彝族 侗族 蒙古族 回族 藏族 维吾尔族 苗族 壮族 朝鲜族 满族 瑶族 白族 土家族 哈尼族 哈萨克族 黎族 " \
              "傈僳族 佤族 畲族 高山族 拉祜族 水族 东乡族 纳西族 " \
              "景颇族 柯尔克孜族 土族 达斡尔族 羌族 撒拉族 毛难族 仫佬族 " \
              "仡佬族 锡伯族 阿昌族 普米族 塔吉克族 怒族 乌孜别克族 " \
              "俄罗斯族 德昂族 保安族 裕固族 崩龙族 独龙族 鄂伦春族 赫哲族 " \
              "门巴族 珞巴族 基诺族 鄂温克族 傣族 京族 塔塔尔族 布朗族 布依族"
# 单字民族对应的相似字符, 用于辅助纠正
similar_data_nation = {
    "侗": ["固", "同", "倜", "调", "垌", "桐", "恫", "洞", "峒", "硐", "胴"],
    "满": ["瞒", "蹒", "螨", "潢", "滿"],
    "汉": ["汶", "汊", "汝", "汐", "汲", "汀", "波", "叹", "仅", "汊"],
    "京": ["惊", "凉", "谅", "掠", "谅", "晾", "掠", "景", "亰"],
    "傣": ["泰", "秦", "倴", "僚", "溙", "奉"],
    "苗": ["亩", "启", "茔", "芸", "盅", "电", "宙", "田", "喵", "描", "猫"],
    "藏": ["臧", "臧", "葬"],
    "壮": ["壯", "莊", "荘", "妆"],
    "瑶": ["摇", "遥", "谣"],
    "白": ["日", "自", "百", "囱", "曰", "囪", "甶", "凹", "汩", "彐",
          "旧", "囗", "田", "帕", "伯", "拍", "泊", "柏", "陌"],
    "黎": ["藜", "棃", "黧", "梨", "犁", "藜"],
    "佤": ["仉", "巩", "讥", "伉", "瓦", "咓", "砙"],
    "畲": ["番", "禽", "肏"],
    "水": ["氷", "囦", "永", "冰", "木", "未"],
    "土": ["工", "二", "三", "王", "亍", "士", "七"],
    "羌": ["恙", "羊", "羔", "羔"],
    "回": ["囚", "四", "迴", "佪", "廻", "洄", "叵", "固", "间", "囧", "区"],
    "怒": ["努", "恕", "奴", "弩", "驽", "孥", "㐐"]
}
# 身份证上面各个元素的准确坐标,长宽,序号,用于判断是否有水印覆盖
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
    "x_d": 85,
    "y_d": 72,
    "w": 24,
    "h": 24,
    "index": 3
}
nationality_ = {
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
    "x_d": 146,
    "y_d": 105,
    "w": 31,
    "h": 23,
    "index": 5
}
birthday_day = {
    "x_d": 193,
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
# 两种水印的尺寸
width_f = height_f = 100
width_c = 177
height_c = 53


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


def judge_nationality(data_n):
    """
    :param data_n: 识别到的民族信息
    :return: 纠正之后的民族
    描述:通过字符匹配程度返回最匹配的民族
    """
    nations = [n[:-1] for n in nationality.split(" ")]
    scores = []  # 各个民族的得分统计list
    if data_n not in nations:
        if len(data_n) == 1:
            # 如果识别到的民族信息没在数据库中,且只有一个字符,则在近似字中搜寻,如果匹配到近似字,则返回对应的民族信息
            for nation in similar_data_nation:
                if data_n in similar_data_nation[nation]:
                    print("民族近似修改:", data_n, "--->", nation)
                    return nation

        for i in range(len(nations)):
            # 通过得分进行纠正
            score = 0
            nation = nations[i]
            for idx_d, k in enumerate(data_n):
                for idx_n, m in enumerate(nation):
                    if k == m:
                        if idx_d == idx_n:
                            score += 100
                        else:
                            score += 50
                        break
            if len(data_n) == len(nation):
                score += 70
            scores.append(score)
    else:
        return data_n
    return nations[scores.index(max(scores))]


def is_overlap(rc1, rc2):
    """
    :param rc1: 矩形框1
    :param rc2: 矩形框2
    :return: 覆盖True, 否则False
    """
    if rc1[0] + rc1[2] > rc2[0] and rc2[0] + rc2[2] > rc1[0] and rc1[1] + rc1[3] > rc2[1] and rc2[1] + rc2[3] > rc1[1]:
        return True
    else:
        return False


def location_check(img_name, template, location_flag, origin_img_path):
    """
    :param img_name: 图片名,包含了水印在这幅图上的具体位置信息,在去水印的时候加上的
    :param template: 校准模板
    :param location_flag: 水印覆盖统计表
    :param origin_img_path: 图片地址
    描述:统计各个元素是否有水印覆盖,更新location_flag
    """
    img_path = os.path.join(origin_img_path, img_name)
    ori_img = cv2.imread(img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
    mark_point = match_img(ori_img, template, 0.4)
    flag_z = True
    mark_point = (max(0, mark_point[0] - 20), mark_point[1])  # 基准坐标
    if img_name.split("_")[1] == "0":  # 根据图片名判断正反面
        flag_z = False
    if mark_point is False:
        print("template match failed")
        return
    if "_fu_" in img_name:  # 禁止复印水印
        points = [int(i) for i in img_name.split("_fu_")[1].split("_")]
        points[0] += 10
        points[1] += 10
        if not flag_z:  # 反面
            # 判断签发机关
            if is_overlap([issuing_unit["x_d"] + mark_point[0], issuing_unit["y_d"] + mark_point[1], issuing_unit["w"],
                           issuing_unit["h"]], [points[0], points[1], width_f, height_f]):
                location_flag[issuing_unit["index"]] = 1
            # 判断有效期限
            if is_overlap(
                    [effective_data["x_d"] + mark_point[0], effective_data["y_d"] + mark_point[1], effective_data["w"],
                     effective_data["h"]], [points[0], points[1], width_f, height_f]):
                if not (is_overlap([effective_data["x_d"] + mark_point[0], effective_data["y_d"] + mark_point[1],
                                    effective_data["w"] / 2, effective_data["h"]],
                                   [points[0], points[1], width_f, height_f]) and
                        is_overlap([effective_data["x_d"] + mark_point[0] + effective_data["w"] / 2,
                                    effective_data["y_d"] + mark_point[1], effective_data["w"] / 2,
                                    effective_data["h"]], [points[0], points[1], width_f, height_f])):  # 不是前后都覆盖
                    if is_overlap([effective_data["x_d"] + mark_point[0], effective_data["y_d"] + mark_point[1],
                                   effective_data["w"] / 2, effective_data["h"]],
                                  [points[0], points[1], width_f, height_f]):  # 覆盖前半部分
                        location_flag[effective_data["index"]] = 1
                    else:  # 覆盖后半部分
                        location_flag[effective_data["index"]] = 2
        else:  # 正面
            # 判断年
            if is_overlap(
                    [birthday_year["x_d"] + mark_point[0], birthday_year["y_d"] + mark_point[1], birthday_year["w"],
                     birthday_year["h"]], [points[0], points[1], width_f, height_f]):
                location_flag[birthday_year["index"]] = 1
            # 判断月
            if is_overlap(
                    [birthday_month["x_d"] + mark_point[0], birthday_month["y_d"] + mark_point[1], birthday_month["w"],
                     birthday_month["h"]], [points[0], points[1], width_f, height_f]):
                location_flag[birthday_month["index"]] = 1
            # 判断日
            if is_overlap([birthday_day["x_d"] + mark_point[0], birthday_day["y_d"] + mark_point[1], birthday_day["w"],
                           birthday_day["h"]], [points[0], points[1], width_f, height_f]):
                location_flag[birthday_day["index"]] = 1
            # 判断地址
            if is_overlap([address["x_d"] + mark_point[0], address["y_d"] + mark_point[1], address["w"],
                           address["h"]], [points[0], points[1], width_f, height_f]):
                location_flag[address["index"]] = 1
            # 判断身份证号
            if is_overlap([id_card["x_d"] + mark_point[0], id_card["y_d"] + mark_point[1], id_card["w"],
                           id_card["h"]], [points[0], points[1], width_f, height_f]):
                location_flag[id_card["index"]] = 1
    if "_chu_" in img_name:  # 复印无效水印,过程如上
        points = [int(i) for i in img_name.split("_chu_")[1].split("_")]
        points[0] += 10
        points[1] += 10
        if not flag_z:  # 反面
            if is_overlap([issuing_unit["x_d"] + mark_point[0], issuing_unit["y_d"] + mark_point[1], issuing_unit["w"],
                           issuing_unit["h"]], [points[0], points[1], width_c, height_c]):
                location_flag[issuing_unit["index"]] = 1
            if is_overlap(
                    [effective_data["x_d"] + mark_point[0], effective_data["y_d"] + mark_point[1], effective_data["w"],
                     effective_data["h"]], [points[0], points[1], width_c, height_c]):
                if not is_overlap([effective_data["x_d"] + mark_point[0], effective_data["y_d"] + mark_point[1],
                                   effective_data["w"] / 2, effective_data["h"]],
                                  [points[0], points[1], width_c, height_c]) and \
                        is_overlap([effective_data["x_d"] + mark_point[0] + effective_data["w"] / 2,
                                    effective_data["y_d"] + mark_point[1], effective_data["w"] / 2,
                                    effective_data["h"]], [points[0], points[1], width_c, height_c]):  # 不是前后都覆盖
                    if is_overlap([effective_data["x_d"] + mark_point[0], effective_data["y_d"] + mark_point[1],
                                   effective_data["w"] / 2, effective_data["h"]],
                                  [points[0], points[1], width_c, height_c]):  # 覆盖前半部分
                        location_flag[effective_data["index"]] = 1
                    else:  # 覆盖后半部分
                        location_flag[effective_data["index"]] = 2
        else:  # 正面
            if is_overlap(
                    [birthday_year["x_d"] + mark_point[0], birthday_year["y_d"] + mark_point[1], birthday_year["w"],
                     birthday_year["h"]], [points[0], points[1], width_c, height_c]):
                location_flag[birthday_year["index"]] = 1
            if is_overlap(
                    [birthday_month["x_d"] + mark_point[0], birthday_month["y_d"] + mark_point[1], birthday_month["w"],
                     birthday_month["h"]], [points[0], points[1], width_c, height_c]):
                location_flag[birthday_month["index"]] = 1
            if is_overlap([birthday_day["x_d"] + mark_point[0], birthday_day["y_d"] + mark_point[1], birthday_day["w"],
                           birthday_day["h"]], [points[0], points[1], width_c, height_c]):
                location_flag[birthday_day["index"]] = 1
            if is_overlap([address["x_d"] + mark_point[0], address["y_d"] + mark_point[1], address["w"],
                           address["h"]], [points[0], points[1], width_c, height_c]):
                location_flag[address["index"]] = 1
            if is_overlap([id_card["x_d"] + mark_point[0], id_card["y_d"] + mark_point[1], id_card["w"],
                           id_card["h"]], [points[0], points[1], width_c, height_c]):
                location_flag[id_card["index"]] = 1


def generate_csv(origin_img_path, json_path, csv_dst_path):
    """
    :param origin_img_path: 图片路径
    :param json_path: 识别结果路径,每套身份证的结果保存在一个json文件中
    :param csv_dst_path: csv文件保存路径
    描述:对识别结果进行纠正,并合成最终的csv文件
    """
    unit_json = "./data_correction_and_generate_csv_file/data/unit.json"  # 签发机关数据库
    id_json = "./data_correction_and_generate_csv_file/data/repitle_address_extract.json"  # 地址数据库
    template_path = "./data_correction_and_generate_csv_file/template_imgs/template_img_2.jpg"  # 模板数据库
    unit_id_json_path = "./data_correction_and_generate_csv_file/data/repitle_idNumber_extract.json"  # 行政区号数据库
    template_img = cv2.imread(template_path, 0)
    # 加载数据
    unit_id_json = json.load(open(unit_id_json_path, "r", encoding="utf-8"))
    id_unit_info = json.load(open(unit_json, "r", encoding="utf-8"))
    for unit_id in unit_id_json:
        if unit_id not in id_unit_info:
            id_unit_info[unit_id] = unit_id_json[unit_id]["unit"]
    if not os.path.exists(csv_dst_path):
        os.makedirs(csv_dst_path)
    files = os.listdir(json_path)
    json_files = [json_name for json_name in files if json_name.endswith(".json")]
    # csv 文件名
    csv_file = "CCFTestResultFixValidData_release.csv"
    # 建立有效日期纠正对象
    valid_data_fixer = ValidData()
    # 建立地址纠正对象
    corrector = AddressCorrect(id_json, unit_json, unit_id_json_path)
    with open(os.path.join(csv_dst_path, csv_file), "w") as fw:
        writer = csv.writer(fw)
        for idx, json_file in enumerate(json_files):
            json_items = json.load(open(os.path.join(json_path, json_file)), encoding="utf-8")
            if len(json_items) < 10:
                # 发现异常,马上退出
                raise Exception("json file except for:%s" % os.path.join(json_path, json_file))
            if idx % 500 == 0:
                print(idx, json_file)
            # 水印覆盖对照表
            context_flag = [i * 0 for i in range(11)]
            # 识别内容存储器
            context = [i for i in range(11)]
            for json_item in json_items:
                img_name = "_".join(json_item.split("_")[1:-1]) + ".jpg"
                data = json_items[json_item]
                jpg_name = json_item.split("_")[1]
                context[0] = jpg_name
                index = int(json_item.split("_")[-1].split(".")[0])
                if index in [1, 9]:
                    # 对正反面各取一张图进行水印覆盖情况检查
                    location_check(img_name, template_img, context_flag, origin_img_path)
                context[index] = data
                if index == 8:
                    if data[-1] == "x":
                        context[8] = data[:-1] + "X"
                if index == 2:
                    # 对民族进行纠正和判断
                    context[2] = judge_nationality(data)
            # 建立出生日期纠正对象
            birth_fix = BirIDNumFix(context[4], context[5], context[6], context[8], context_flag)
            # 对出生日期和身份证信息进行判断和纠正
            context[4], context[5], context[6], context[8] = birth_fix.fix_birthday()
            # 修改性别
            if context[3] not in ["男", "女"]:
                if context[3] in ["文", "安", "仗", "如", "汝", "奴", "义", "丈", "好", "乂", "㚢", "囡"]:
                    context[3] = "女"
                else:
                    context[3] = "男"
            # 修改有效期
            context[10] = valid_data_fixer.fix_valid_data(context[10], context)
            # 修改签证机关和身份证前6位
            context[8], context[9] = corrector.unit_correct(context[4], context[8], context[9], context[7],
                                                            context_flag)
            # 修改地址
            if "-" in context[7]:
                context[7] = "".join(context[7].split("-"))
            context[7] = corrector.address_correct(context[7], context[8][:6])
            id_new = ""
            for i in context[8]:
                if i in "0123456789X":
                    id_new += i
                else:
                    print("身份证异常字符:", context[8])  # 过滤身份证中的异常字符
            context[8] = id_new
            # 向csv文件中写入一条数据
            writer.writerow(context)

