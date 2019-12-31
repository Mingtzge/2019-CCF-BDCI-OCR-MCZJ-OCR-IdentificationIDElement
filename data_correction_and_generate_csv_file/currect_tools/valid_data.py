import sys
import copy


def judge_str(a, b):
    """
    :param a: 字符串a
    :param b: 字符串b
    :return: 数组
    描述:比较字符串a和字符串b那些位是不同的
    """
    dif_seq = []
    for i in range(len(a)):
        try:
            if a[i] != b[i]:
                dif_seq.append(i)
        except IndexError:
            print("invalid a:%s  b%s" % ("".join(a), "".join(b)))
            return []
    return dif_seq


class ValidData:
    """
    描述:为了避免干扰,在识别结果中没有加入小数点的识别,所以对于有效期的识别结果需要转化.
    识别格式:20191103-20291103
    转化格式:2019.11.03-2029.11.03
    对于识别结果存在异常的,会进行纠正.
    根据身份证有效期规则:年龄16岁以下,有效期5年,16-26岁,有效期10年,26-46岁,有效期20年,大于46岁,有效期为长期  进行纠正
    """

    def __init__(self, _default_month="0", _default_day="1"):
        self.default_month = _default_month
        self.default_day = _default_day

    def fix_month(self, a, b=None):
        """
        :param a: 月份a
        :param b: 月份b,有值时与月份a相互纠正,默认以a为基准
        :return: 纠正之后的值
        描述:主要是保证月份的数值在0-12之间
        """
        if b is None:
            if a not in "01":
                a = self.default_month
            return a
        else:
            if a not in "01" and b in "01":
                a = b
            elif a in "01" and b not in "01":
                b = a
            elif a not in "01" and b not in "01":
                a = b = self.default_month
            return a, b

    def fix_day(self, a, b=None):
        """
        :param a: 日a
        :param b: 日b,有值时与日a相互纠正,默认以a为基准
        :return: 纠正之后的值
        描述:主要是保证日的数值在0-31之间
        """
        if b is None:
            if a not in "0123":
                a = self.default_day
            return a
        else:
            if a not in "0123" and b in "0123":
                a = b
            elif a in "0123" and b not in "0123":
                b = a
            elif a not in "0123" and b not in "0123":
                a = b = self.default_day
            return a, b

    def fix_valid_data(self, data, context):
        """
        :param data: 识别的有效日期
        :param context: 识别的所有元素所在的数组
        :return: 转化和纠正之后的有效期
        """
        valid = data.split("-")
        start_data = list(valid[0])  # 起始值
        if len(valid) < 2:
            # 异常情况,当没有识别到"-"时,按位数进行切割
            if len(start_data) > 7:
                end_data = start_data[8:]
                start_data = start_data[:8]
            else:
                end_data = start_data
        elif len(valid) == 2:
            end_data = list(valid[1])
            if len(start_data) > 8:
                # 当字符长度不对时,进行处理
                if len(end_data) == 8:
                    if valid[1][4:] in valid[0]:
                        str_data = valid[0].split(valid[1][4:])[0]
                        if len(str_data) >= 4:
                            start_data = list(str_data[:4] + valid[1][4:])
        elif len(valid) > 2:
            # 取长度更接近的一项
            a = [abs(len(b) - 8) for b in valid[1:]]
            end_data = list(valid[a.index(min(a))])
        if len(start_data) < 8:
            if len(start_data) > 4:
                if len(end_data) == 8 and "_".join(end_data[4:]) not in "_".join(start_data):  # 月日缺失
                    start_data = start_data[:4] + end_data[4:]
                else:
                    start_data = list("201") + start_data[len(start_data) - 5:]  # 年缺失
            else:
                start_data = list("20199999")  # 无法纠正, 投降~
        if start_data[0:3] != list("201"):  # 日期基本上都是2000年之后的,强行检测
            if start_data[0:3] == list("200"):
                start_data = list("2009") + start_data[4:]
            else:
                start_data = list("201") + start_data[3:]
                print("except valid_start_time for %s " % start_data)
        age_diff = 30  # 起始日期时的年龄
        try:
            age_diff = int("".join(start_data[:4])) - int(context[4])  # 获取办证时的年龄,用于纠正起始日期和终止日期的跨度
        except Exception:
            print("exe_error: %s" % sys.exc_info()[0])
            pass
        seg_year = 1  # 跨度,单位10年
        if age_diff >= 26:
            seg_year = 2
        if "长" in end_data or "期" in end_data or (age_diff > 46 and len(end_data) < 7):
            end_data = list("长期")
            start_data[4] = self.fix_month(start_data[4])  # 纠正月
            start_data[6] = self.fix_day(start_data[6])  # 纠正日
            end_data = ''.join(end_data)
            for idx_s, i in enumerate(start_data):
                if i not in "0123456789":
                    start_data[idx_s] = "8"  # 对于异常值,强行更改
                    print("invalid word for start:%s  fix to %s" % (
                        "".join(start_data), start_data[idx_s]))
        else:
            for idx_s, i in enumerate(start_data):  # 纠正起始日期的异常值,用终止日期替代
                if i not in "0123456789":
                    if len(end_data) > idx_s:
                        start_data[idx_s] = end_data[idx_s]
                    else:
                        start_data[idx_s] = "5"
                    print("invalid word for start:%s  end:%s   fix to %s" % (
                        "".join(start_data), "".join(end_data), start_data[idx_s]))
            for idx_e, i in enumerate(end_data):  # 纠正终止日期的异常值,用起始日期替代
                if i not in "0123456789":
                    if len(start_data) > idx_e:
                        end_data[idx_e] = start_data[idx_e]
                    else:
                        end_data[idx_e] = "5"
                    print("invalid word for end:%s  start:%s   fix to %s" % (
                        "".join(end_data), "".join(start_data), end_data[idx_e]))
            if len(end_data) == 8:
                end_data = list("20") + end_data[2:]
                if not int("".join(end_data[:4])) - int("".join(start_data[:4])) in [5, 10, 20]:  # 检测时间跨度是否正常
                    diff_idx = judge_str(start_data, end_data)  # 判断那些位不一样
                    if 2 in diff_idx:  # 纠正年的十位
                        if int(end_data[2]) - int(start_data[2]) > 2:
                            end_data[2] = str(int(start_data[2]) + seg_year)
                    else:
                        if age_diff < 16:
                            y_ = int(start_data[3]) + 5
                            shi = int(y_ / 10)
                            end_data[3] = str(int(start_data[3]) + y_ % 10)
                            if shi == 1:
                                end_data[2] = str(int(start_data[2]) + 1)
                        else:
                            end_data[2] = str(int(start_data[2]) + seg_year)
                start_data[4], end_data[4] = self.fix_month(start_data[4], end_data[4])  # 纠正月
                start_data[6], end_data[6] = self.fix_day(start_data[6], end_data[6])  # 纠正日
                for loc in [5, 7]:  # 纠正月和日的个位,强行与起始日期保持一致
                    if loc == 7:
                        if not (end_data[4:6] == start_data[4:6] == list("02") and (
                                "".join(start_data[6:]) in ["28", "29"] and "".join(end_data[6:]) in ["28", "29"])):
                            end_data[loc] = start_data[loc]  # 排除润年的情况
                    else:
                        end_data[loc] = start_data[loc]
                if int("".join(end_data[:4])) - int("".join(start_data[:4])) != 5:  # 纠正年的个位
                    end_data[3] = start_data[3]
            else:  # 当终止日期位数不对时,强行以起始日期为基准修改
                end_data = list("20") + end_data[2:]
                if int("".join(end_data[:4])) - int("".join(start_data[:4])) in [5, 10, 20]:
                    end_data = end_data[:4] + start_data[4:]
                else:
                    end_data = copy.copy(start_data)
                    end_data[2] = str(int(start_data[2]) + seg_year)
            end_data = ''.join(end_data)
            end_data = end_data[:4] + "." + end_data[4:6] + "." + end_data[6:]
        start_data = ''.join(start_data)
        start_data = start_data[:4] + "." + start_data[4:6] + "." + start_data[6:]
        return start_data + "-" + end_data
