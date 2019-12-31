import json
import copy


def str_recommends(dict, data, adapt=False):
    """
    :param dict: list或者字符串
    :param data: 字符串,用于与dict中的每一项作匹配
    :param adapt: 标识,True,dict为字符串,计算dict与data的相似度, False,dict为list, 搜寻最匹配的推荐项返回(如果满足要求)或者data本身
    :return: 如果adapt为True,返回data与匹配度最高项的相似度得分,adapt默认是False,返回推荐值(如果满足要求)或者data本身
    描述: 这是一个字符串推荐或者打分函数,通过将data中的每一个字与各个推荐项的每一个字进行比对,根据打分规则,对各个推荐项进行打分,分数最高的项
    即为推荐项.  比如:dict["2345", "6789", "9842"] ,data:"234" 推荐就为dict中的"2345"
    作用:用于辅助签发机关,身份证前六位,和地址各级单位与识别结果纠正
    打分规则:
    字符但位置不同相同equl_:7分
    字符且位置相同loc_equl_:15分
    长度相同len_equl:30分
    """
    scores = []
    loc_equl_ = 15
    equl_ = 7
    len_equl = 30
    if adapt:  # 如果是计算相似度得分,降低长度相同时的得分
        len_equl = 15
    if len(dict) == 0:
        if adapt:
            print("推荐异常:", dict, "<-------->", data)
            return 0
        return data
    data_bak = copy.copy(data)  # 保存data数据
    if not adapt:  # 搜寻匹配项
        potions = [p for p in dict]
        for potion in potions:
            data = copy.copy(data_bak)
            score = 0
            score1 = 0
            flag = False
            if potion == "":
                scores.append(score)
                continue
            if potion[-2:] in data and len(data.split(potion[-2:])) == 2 and potion[-1] == "镇":
                # 当是匹配镇级单位时, 切除村的部分,避免造成干扰
                data = data.split(potion[-2:])[0] + potion[-2:]
            if potion[-1] in ["镇", "村"] and potion[-1] in data and len(data) >= len(potion) > 3:
                #  通常这种情况是最后一级比对, 剔除推荐项中的最后一位,减少干扰
                new_data = ""
                data_bak = copy.copy(data)
                for i in data:
                    if i not in "-1234567890":
                        new_data += i  # 剔除data中的无效字符
                data = new_data
                if len(potion) <= len(data):
                    #  剔除前要更改得分情况
                    if potion[-1] == data[len(potion) - 1]:
                        score += loc_equl_
                    else:
                        score += equl_
                    if len(potion) == len(data):
                        score += len_equl
                    potion = potion[:-1]
                    flag = True
                else:
                    print("长度发生变化:", data_bak, "------>", data)
            if len(data) < len(potion) or "村" in potion:
                #  从右向左计算得分
                p = potion[::-1]
                d = data[::-1]
                for idx_d, k in enumerate(d):
                    default_len = len(d)
                    if len(data) > len(potion):
                        default_len = len(potion)
                    elif len(potion) - len(data) < 2:
                        default_len = len(potion)  # 避免小范围漏字
                    for idx_n, m in enumerate(p[:default_len]):
                        if k == m:
                            if idx_d == idx_n:
                                score1 += loc_equl_
                            else:
                                score1 += equl_
                            break
            for idx_d, k in enumerate(potion):
                #  从左向右计算得分
                for idx_n, m in enumerate(data[:min(len(potion), len(data))]):
                    if k == m:
                        if idx_d == idx_n:
                            score += loc_equl_
                        else:
                            score += equl_
                        break
            #  计算长度相等时的得分
            if flag:
                if len(potion) - len(data) - 1 == 0:
                    score += equl_
            elif len(potion) - len(data) == 0:
                score += equl_
            scores.append(max(score, score1))
        score = max(scores)
        recommend_str = potions[scores.index(score)]  # 取出得分最高的项
        if score > int(len(recommend_str) * 8 * 0.7):  # 当匹配程度大于70%时，才可靠推荐
            return potions[scores.index(max(scores))]
        else:
            return data  # 否则,返回原始数据
    else:
        '''
        进行相似度计算
        主要是将匹配度得分与理想得分做比值, 匹配度得分与上面计算方式一致,但是步骤要简单些,主要是针对与签发机关使用
        '''
        option = dict
        origin_score = len(data) * loc_equl_ + len_equl   # 完全相等时的理想得分
        score = 0
        if "公安局" in option:
            option = option[:-3]  # 剔除公有的字符,避免造成影响
        if len(dict) == len(data):
            for i in range(len(dict)):
                k = dict[i]
                m = data[i]
                if k == m:
                    score += loc_equl_
        else:
            for idx_d, k in enumerate(option):
                for idx_n, m in enumerate(data[:min(len(option), len(data))]):
                    if k == m:
                        if idx_d == idx_n:
                            score += loc_equl_
                        else:
                            score += equl_
                        break
        if len(option) == len(data):
            score += len_equl
        return score / origin_score


class AddressCorrect:
    """
    :param label_json:全国地址数据库,具体到村级,70万+
    :param unit_json:全国签发机关数据库
    :param id_num_json:全国区号数据库,每一个区号对应一个前三级地址和身份证签发机关单位
    描述:这个类的作用主要是根据整理到全国的各地区的详细地址,行政区号.签发机关,对我们的识别结果进行纠正.
    纠正范围:签发机关,地址,身份证前六位
    """
    def __init__(self, label_json, unit_json, id_num_json):
        #  加载数据,对数据进行一定格式地处理
        self.label_data = json.load(open(label_json, "r", encoding="utf-8"))
        self.unit_json = json.load(open(unit_json, "r", encoding="utf-8"))
        self.idNumber_json = json.load(open(id_num_json, "r", encoding="utf-8"))
        self.unit_id = {}
        self.address_id = {}
        self.new_address_info = {}
        for unit_item in self.unit_json:
            self.unit_id[self.unit_json[unit_item]] = unit_item
        for address_item in self.idNumber_json:
            self.address_id[self.idNumber_json[address_item]["address"]] = address_item
        for level1 in self.label_data:
            for level2 in self.label_data[level1]:
                for level3 in self.label_data[level1][level2]:
                    self.new_address_info[level1 + level2 + level3] = self.label_data[level1][level2][level3]
        self.default_len = 4  # 切割单元地址的默认长度

    def unit_correct(self, year, id_number, unit_address, address, context_flag):
        """
        :param year: 识别到的年份,用于切割识别到的身份证的行政区号
        :param id_number: 识别到的身份证
        :param unit_address: 识别到的签证机关
        :param address: 识别到的地址
        :param context_flag: 水印位置标识,里面记录了身份证上面每个元素上是否可能有水印覆盖
        :return: 返回纠正之后的身份证号和签证机关
        """
        #  记录各个元素是否有水印
        id_flag = context_flag[8] == 1
        unit_flag = context_flag[9] == 1
        address_flag = context_flag[7] == 1
        flag = False
        if id_flag and not unit_flag:
            #  如果身份证有水印覆盖签发机关没有水印覆盖
            if unit_address in self.unit_id:
                #  当签发机关在数据库中存在时, 识别错误的概率极低,此时不用纠正
                #  获取这个签发机关对应的行政区号(身份证前6位)
                rec_id = self.unit_id[unit_address]
                if rec_id != id_number.split(year)[0]:
                    # 与识别到的身份证年份前的数据进行判断, 如果不相等,则认为身份证识别出错,进行替换
                    print("身份证前6位识别错误!修改身份证号:", id_number, "----->",
                          rec_id + year + year.join(id_number.split(year)[1:]))
                    id_number = rec_id + year + year.join(id_number.split(year)[1:])
                    print("身份证前6位识别错误!修改签发机关:", unit_address, "----->", self.unit_json[rec_id])
                    unit_address = self.unit_json[rec_id]
                return id_number, unit_address
        if id_number.split(year)[0] in self.unit_json:
            #  当身份证前六位在数据库存在时
            if not id_flag and unit_flag:
                # 如果身份证没有水印覆盖,而签发机关有, 直接更换签发机关
                unit_address = self.unit_json[id_number.split(year)[0]]
            if not unit_flag and id_flag and unit_address in self.unit_id:
                # 如果身份证有水印覆盖, 签发机关没有水印覆盖, 签发机关在数据库中存在时, 直接更换身份证前6位
                id_number = self.unit_id[unit_address] + year + year.join(id_number.split(year)[1:])
            #  计算识别到的签发机关和身份证前六位对应的签发机关的相似度
            similar_index = str_recommends(self.unit_json[id_number[:6]], unit_address[:-3], adapt=True)
            if similar_index < 0.4 or unit_address in self.unit_id:
                # 当相似度低于0.4 或者识别到的签发机关也在数据库中,进行更进一步的判断
                # 获得推荐的签发机关
                str_ = str_recommends(self.unit_id, unit_address[:-3])
                if unit_address in self.unit_id:
                    # 当签发机关就在数据库中,不采用推荐的,这种情况两者应该相等,保险一点~
                    str_ = unit_address
                if id_number[:6] in self.idNumber_json:
                    # 通过地址信息辅助判断当前身份证前六位的置信度
                    if self.idNumber_json[id_number[:6]]["address"] in address:
                        flag = True
                if not flag:
                    if str_ in self.unit_id:
                        rec_id = self.unit_id[str_]
                        if rec_id != id_number[:6]:  # 可能是身份证前6位识别错误,修改身份证号
                            print("身份证前6位识别错误!修改身份证号:", id_number, "----->",
                                  rec_id + year + year.join(id_number.split(year)[1:]))
                            id_number = rec_id + year + year.join(id_number.split(year)[1:])
                            print("身份证前6位识别错误!修改签发机关:", unit_address, "----->", self.unit_json[rec_id])
                            unit_address = self.unit_json[rec_id]
                            return id_number, unit_address
            elif id_number[:6] in self.idNumber_json and similar_index < 0.8 and not address_flag:  # 地址相似度校验
                #  追加一层校验,加入地址信息,确保纠正的正确性
                level3 = self.idNumber_json[id_number[:6]]["address"]
                similar_index = str_recommends(level3, address[:len(level3)], adapt=True)
                if similar_index < 0.65:
                    rec_ad = str_recommends(self.address_id, address)
                    if rec_ad in self.address_id:
                        similar_index = str_recommends(id_number[:6], self.address_id[rec_ad], adapt=True)
                        if similar_index > 0.8:
                            print("身份证前6位识别错误!修改身份证号:", id_number, "----->",
                                  self.address_id[rec_ad] + year + year.join(id_number.split(year)[1:]))
                            id_number = self.address_id[rec_ad] + year + year.join(id_number.split(year)[1:])
                            if self.address_id[rec_ad] in self.unit_json:
                                print("身份证前6位识别错误!修改签发机关:", unit_address, "----->",
                                      self.unit_json[self.address_id[rec_ad]])
                                unit_address = self.unit_json[self.address_id[rec_ad]]
                            return id_number, unit_address
            unit_address = self.unit_json[id_number[:6]]
        else:
            #  当身份证在数据库中不存在时
            if id_flag:
                #  如果有水印覆盖,可以肯定是识别出错了, 然后进行纠正,过程如上
                str_ = str_recommends(self.unit_id, unit_address)
                similar_index = 0
                if len(str_) > 1 and str_ in self.unit_id:
                    similar_index = str_recommends(self.unit_id[str_], id_number.split(year)[0], adapt=True)
                if similar_index > 0.5:
                    print("缺失!修改身份证号:", id_number, "----->",
                          self.unit_id[str_] + year + year.join(id_number.split(year)[1:]))
                    id_number = self.unit_id[str_] + year + year.join(id_number.split(year)[1:])
                    print("缺失!修改签发机关:", unit_address, "----->", str_)
                    unit_address = str_
                else:
                    str_ = str_recommends(self.unit_json, id_number[:6])
                    similar_index = str_recommends(self.unit_json[str_], unit_address[:-3], adapt=True)
                    if similar_index > 0.5:
                        print("缺失!修改身份证号:", id_number, "----->", str_ + year + year.join(id_number.split(year)[1:]))
                        id_number = str_ + year + year.join(id_number.split(year)[1:])
                        print("缺失!修改签发机关:", unit_address, "----->", self.unit_json[str_])
                        unit_address = self.unit_json[str_]
        new_id = []
        for i in id_number:
            if i in "0123456789X":
                new_id.append(i)  # 过滤身份证中的无效字符
        id_number = "".join(new_id)
        return id_number, unit_address

    def address_correct(self, address_data, id_num):
        """
        :param address_data: 识别到的地址数据
        :param id_num: 识别到的身份证数据
        :return: 纠正之后的地址信息
        描述:纠正可能识别错的地址数据
        """
        #  加载地址数据,一个5级 省 市 县/区/旗 镇 村
        items = self.label_data
        in_flag = True
        level = 0
        str_ = ""
        if id_num in self.idNumber_json:
            #  如果身份证前六位在数据库中,直接提取对应的前三级地址
            level = 4
            items = self.new_address_info
            level3 = self.idNumber_json[id_num]["address"]
            if level3[-2:] in address_data:
                #  替换掉识别数据的对应部分
                if abs(len(level3) - len(address_data.split(level3[-2:])[0])) < 4:
                    address_data = level3 + level3[-2:].join(address_data.split(level3[-2:])[1:])
                else:
                    address_data = level3 + address_data[len(level3):]
            self.default_len = len(level3)
            if level3 in self.new_address_info:
                items = self.new_address_info[level3]  # 将地址索引到第三级,地址范围缩小到镇级别
                str_ = level3
        while len(items) > 0:
            #  循环,直到纠正到最后一级
            if str_ == "":
                address_split = address_data[:self.default_len]
            else:
                #  切割识别到地址数据,对其余数据进行纠正
                if str_[-2:] in address_data and len(address_data.split(str_[-2:])) == 2:
                    address_split = address_data.split(str_[-2:])[-1]
                else:
                    address_split = address_data[len(str_):len(address_data)]
            #  在本级地址范围内,搜寻可能匹配的地址信息
            recommend_data = str_recommends(items, address_split)
            if level == 4 and len(address_split) > 0 and recommend_data in items:
                #  当是第四级(镇)时, 判断推荐信息的可信度,如果可信度不高,增加村级的判断
                try:
                    similar_index = str_recommends(recommend_data,
                                                   address_split[:min(len(recommend_data), len(address_split))],
                                                   adapt=True)
                    if similar_index < 0.87:
                        if recommend_data[-2:] in address_split and len(address_split.split(recommend_data[-2:])) == 2:
                            if len(address_split.split(recommend_data[-2:])[-1]) > 0:
                                split_data = address_split.split(recommend_data[-2:])[-1]
                        else:
                            split_data = address_split[len(recommend_data):]
                        if split_data not in items[recommend_data]:
                            in_flag = False  # 可信标记
                    level = 0
                except Exception:
                    print("第四级纠正异常:", recommend_data, address_split)
            if recommend_data not in items or in_flag is False:
                # 本层没有合适的或者可信度不高,向下搜索一层,来决定这一层采用什么值
                if isinstance(items, dict):
                    flag = False
                    for item in items:
                        if item in address_data[len(str_):]:
                            recommend_data = item
                            break
                        for dd in items[item]:
                            if dd in address_data[len(str_):] and len(dd) > 1:
                                #  如果下一级单元在地址中,则认为其父级的可信度高
                                similar_index = str_recommends(item,
                                                               address_split[:min(len(item), len(address_split))],
                                                               adapt=True)
                                # 判断父级的相似度, 如果满足要求, 则采用父级
                                if similar_index < 0.5:
                                    break
                                str_ += item
                                items = items[item]
                                recommend_data = dd
                                flag = True
                                break
                        if flag:
                            break
                in_flag = True
            # 相加地址单元
            str_ += recommend_data
            if (len(str_) >= len(address_data) or isinstance(items, list)) or recommend_data not in items:
                return str_  # 返回纠正之后的数据
            items = items[recommend_data]  # 地址指针下移
            level += 1
