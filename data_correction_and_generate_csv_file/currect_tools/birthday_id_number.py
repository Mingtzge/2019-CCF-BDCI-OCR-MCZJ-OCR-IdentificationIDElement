class BirIDNumFix:
    def __init__(self, year, month, day, id_num, context_flag):
        """
        :param year: 识别的年
        :param month: 识别的月
        :param day: 识别的日
        :param id_num: 识别的身份证号
        :param context_flag: 水印位置标记
        描述:身份证中也是包含年月日的,可以与身份证上面的年月日进行相互纠正.
        """
        self.year_flag = context_flag[4] == 1
        self.month_flag = context_flag[5] == 1
        self.day_flag = context_flag[6] == 1
        self.id_flag = context_flag[8] == 1
        self.year_19 = list("19")
        self.year_20 = list("20")
        self.year = list(str(year))
        self.day = list(str(day))
        self.month = list(str(month))
        if self.year_flag and not self.id_flag:
            self.year = list(id_num[6:10])  # 当年有水印覆盖时,采用身份证上面的数据
        if self.month_flag and not self.id_flag:
            self.month = list(id_num[10:12])  # 当月有水印覆盖时,采用身份证上面的数据
        if self.day_flag and not self.id_flag:
            self.day = list(id_num[12:14])  # 当日有水印覆盖时,采用身份证上面的数据
        # 检验年月日的位数情况,进行转化,如将2019 1 1转化为2019 01 01
        if len(self.year) != 4:
            if not self.id_flag:
                self.year = list(id_num[6:10])
            elif len(self.year) > 2 and self.year[:2] != self.year_19 and self.year[:2] != self.year_20:
                self.year = self.year_19 + self.year[-2:]
            else:
                self.year = list("2012")
                print("年没有找到合适的匹配!!!")
        if len(self.month) == 1:
            self.month = list("0" + month)
        elif len(self.month) == 0:
            if not self.id_flag:
                self.month = list(id_num[10:12])
            if year in id_num and len(year) == 4:
                self.month = list(id_num.split(year)[1][0:2])
            elif len(id_num) > 14:
                self.month = list(id_num[10:12])
            else:
                print("月没有找到合适的匹配!!!")
                self.month = list("01")
        if len(self.day) == 1:
            self.day = list("0" + day)
        elif len(self.day) == 0:
            if not self.id_flag:
                self.day = list(id_num[12:14])
            if year in id_num and len(year) == 4:
                self.day = list(id_num.split(year)[1][2:4])
            elif len(id_num) > 14:
                self.day = list(id_num[12:14])
            else:
                print("日没有找到合适的匹配!!!")
                self.day = list("01")
        self.id_num = list(id_num)
        self.id_num_s = id_num
        self.y_m_d = self.year + self.month + self.day
        self.id_len = len(self.id_num)
        self.valid_bir = True

    def judge_birth(self, date):
        """
        :param date: 生日信息
        :return: 更新self.valid_bir标志位,检测是否正常
        """
        if len(date) < 8:
            self.valid_bir = False
            return
        if len(date) > 8:
            self.valid_bir = False
            return
        for i in date:
            if i not in "0123456789":
                self.valid_bir = False
                return
        if (date[:2] != self.year_19 and date[:2] != self.year_20) or date[4] not in "01":
            self.valid_bir = False
            return
        if date[6] not in "0123":
            self.valid_bir = False
            return
        if date[6] == "3":
            if date[7] not in "01":
                self.valid_bir = False

    def correct_birth(self, date1, date2):
        """
        :param date1: 身份证上的出生日期
        :param date2: 身份证中的出生日期
        描述:二者相互纠正,前者有异常就用后者替代,反之同理,当都存在异常值时,采用默认值
        """
        if len(date1) < 8:
            for i in range(8 - len(date1)):
                date1.append("0")
        if len(date2) < 8:
            for i in range(8 - len(date2)):
                date2.append("0")
        for index, i in enumerate(date1):
            if date1[index] not in "0123456789" and date2[index] in "0123456789":
                date1[index] = date2[index]
            elif date2[index] not in "0123456789" and date1[index] in "0123456789":
                date2[index] = date1[index]
            elif date2[index] not in "0123456789" and date1[index] not in "0123456789":
                date2[index] = date1[index] = "0"
        if (date1[:2] != self.year_19 and date1[:2] != self.year_20) and (
                date2[:2] != self.year_19 and date2[:2] != self.year_20):
            date1[:2] = date2[:2] = list("19")
        elif date1[:2] != self.year_19 and date1[:2] != self.year_20:
            date1[:2] = date2[:2]
        else:
            date2[:2] = date1[:2]
        # 月
        if date1[4] not in "01" and date2[4] in "01":
            date1[4] = date2[4]
        elif date2[4] not in "01" and date1[4] in "01":
            date2[4] = date1[4]
        elif date2[4] not in "01" and date1[4] not in "01":
            date2[4] = date1[4] = "0"
        # 日
        if date1[6] not in "0123" and date2[6] in "0123":
            date1[6] = date2[6]
        elif date2[6] not in "0123" and date1[6] in "0123":
            date2[6] = date1[6]
        elif date2[6] not in "0123" and date1[6] not in "0123":
            date2[6] = date1[6] = "0"
        if date1[6] == "3":
            if date1[7] not in "01" and date2[7] in "01":
                date1[7] = date2[7]
            else:
                date1[7] = "0"
        if date2[6] == "3":
            if date2[7] not in "01" and date1[7] in "01":
                date2[7] = date1[7]
            else:
                date2[7] = "0"

    def fix_birthday(self):
        """
        :return: 年 月 日 身份证号
        """
        if "".join(self.y_m_d) in "".join(self.id_num):  # 可靠性高， 不用纠正
            return "".join(self.year), str(int("".join(self.month))), str(int("".join(self.day))), "".join(self.id_num)
        else:
            start_loc = 6  # 定位身份证出生日期起始位置， 粗略定位，用年、日月
            if "".join(self.year) in self.id_num_s and len(self.year) == 4:
                start_loc = len(self.id_num_s.split("".join(self.year))[0])
            elif "".join(self.month + self.day) in self.id_num_s:
                start_loc = len(self.id_num) - len(self.id_num_s.split("".join(self.month + self.day))[-1]) - 8
            self.judge_birth(self.y_m_d)
            if not self.valid_bir:  # 出生日期存在问题
                self.valid_bir = True
                self.judge_birth(self.id_num[start_loc:start_loc + 8])  # 判断身份证上面的信息有没有问题
                if self.valid_bir:  # 身份证信息没有问题, 直接替换
                    y_m_d = "".join(self.id_num[start_loc:14])
                    # print("fix  after:",y_m_d[:4], str(int(y_m_d[4:6])), str(int(y_m_d[6:])), "".join(self.id_num))
                    return y_m_d[:4], str(int(y_m_d[4:6])), str(int(y_m_d[6:])), "".join(self.id_num)
                else:  # 身份证信息存在问题
                    self.correct_birth(self.y_m_d, self.id_num[start_loc:start_loc + 8])  # 二者相互纠正
                    for index, i in enumerate(self.y_m_d):
                        if i not in "0123456789":
                            self.y_m_d[index] = "0"
                    for index, i in enumerate(self.id_num):
                        if i not in "0123456789":
                            self.id_num[index] = "0"
                    y_m_d = "".join(self.y_m_d)
                    return y_m_d[:4], str(int(y_m_d[4:6])), str(int(y_m_d[6:])), "".join(self.id_num)
            else:  # 出生日期没有问题
                self.valid_bir = True
                self.judge_birth(self.id_num[start_loc:start_loc + 8])  # 判断身份证上面的信息有没有问题
                self.id_num[start_loc:start_loc + 8] = self.y_m_d
                return "".join(self.year), str(int("".join(self.month))), str(int("".join(self.day))), "".join(
                    self.id_num)
