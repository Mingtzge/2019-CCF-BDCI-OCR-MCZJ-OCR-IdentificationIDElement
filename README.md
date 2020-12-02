
# 整体介绍
  [赛题介绍](https://www.datafountain.cn/competitions/346)
  
  
  我们的队名是:鹏脱单攻略队  后面改为"天晨破晓" 最终我们团队成绩在复赛AB榜均排在第一名,识别准确率达0.996952
 
  ![官网截图](https://github.com/Mingtzge/2019-CCF-BDCI-OCR-MCZJ-OCR-IdentificationIDElement/blob/master/show_imgs/webwxgetmsgimg.jpg)
  
  团队成绩:2019CCF-BDCI大赛  最佳创新探索奖 和 "基于OCR的身份证要素提取"单赛题冠军
  
  ### 系统处理流程图
  
  ![流程图](https://github.com/Mingtzge/2019-CCF-BDCI-OCR-MCZJ-OCR-IdentificationIDElement/blob/master/show_imgs/%E7%B3%BB%E7%BB%9F%E6%9E%B6%E6%9E%84.png)
  
  ### 方案亮点 
  我们采用条件生成对抗网络(CGAN)处理赛题中的水印干扰,取到了比较好的效果,展示一下效果图片:
  ![去水印效果](https://github.com/Mingtzge/2019-CCF-BDCI-OCR-MCZJ-OCR-IdentificationIDElement/blob/master/show_imgs/%E5%8E%BB%E6%B0%B4%E5%8D%B0%E6%95%88%E6%9E%9C.png)
  
  [生成仿真数据源码](https://github.com/Mingtzge/2019-CCF-BDCI-OCR-MCZJ-fake_data_generator),生成仿真训练数据训练去水印模型和文字识别模型
  
  [方案PPT](https://discussion.datafountain.cn/questions/2260/answers/23380) [方案论文](https://discussion.datafountain.cn/questions/2232/answers/23321)
  
# 执行方式介绍
    完整执行示例:
    CPU执行,单进程:
    python main_process.py --test_experiment_name test_example --test_data_dir ./test_data --gan_ids -1 --pool_num 0
    参数详解:
    --test_experiment_name:实验名,将决定中间数据结果存放目录
    --test_data_dir: 数据目录
    --gan_ids: 去水印模型:如果是-1 则是cpu运行, 大于0,则是GPU
    --pool_num 0单进程 大于0多进程
    其他参数参考main_process.py中的help

# 项目整体文件结构说明:
(按照处理流程介绍,具体文件介绍见文件内的readme)

## 身份证区域提取模块 cut_twist_process
  剪切、翻转部分代码；用于将身份证正反面从原始图片中切分出来

## 去除水印\关键文本定位模块 watermask_remover_and_split_data
  进行水印去除,身份证切割,提取文字部分,滤波

## 去水印模型 pytorch-CycleGAN-and-pix2pix
  我们训练好的去除水印模型地址:
### 参考资料
  去水印模型采用条件gan网络。论文[链接](https://arxiv.org/pdf/1611.07004.pdf)
  
  参考了GitHub上gan pix2pix 项目，[链接](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix#cyclegan-and-pix2pix-in-pytorch),我们基于此项目进行了一些更改

## 文字识别模块 recognize_process
  用于识别图片中的文字信息
### 参考资料
  识别模型采用CRNN。论文[链接](https://arxiv.org/abs/1507.05717)
  
  参考了GitHub上两个模型的TensorFlow实现
  
  [项目1](https://github.com/MaybeShewill-CV/CRNN_Tensorflow) 
  
  [项目2](https://github.com/bai-shang/crnn_ctc_ocr.Tensorflow)

## 文本纠正模块 data_correction_and_generate_csv_file
  对识别结果进行纠正,以及生成最终的csv文件

## data_temp
  中间数据存放目录,目录名称:实验名+日期

## CCFTestResultFixValidData_release.csv
  生成的结果文件

## main_process.py
  执行脚本

## Requirement.txt
  运行的环境要求

# 注意
  去水印模型[地址](https://github.com/Mingtzge/models_data)
  
  这项是工程的submodel,需要克隆的时候需要加上"--recursive"参数 
  去水印模型较大,采用了git-lfs,要安装这个包,不然可能会导致clone失败或者比较慢
  
  
  由于lfs超出限额，功能失效，可能会导致模型文件（文字识别和去水印模型）clone失败，我们把模型文件上传百度云了，由于数据文件包含身份证图片，数据敏感，百度云链接很容易失效，需要赛题数据和模型文件的话，添加我百度网盘好友，具体见[About data download](https://github.com/Mingtzge/2019-CCF-BDCI-OCR-MCZJ-OCR-IdentificationIDElement/issues/10#issue-546035222)

!!!注:测试数据跟初赛和复赛的数据格式需要保持一致,每面身份证左上角需要有:"仅限DBCI比赛(复赛)使用"字样,
      且字体大小格式位置应该跟初赛和复赛的保持一致,否则将严重影响识别的准确性甚至代码运行出错
   原因:对于身份证各个元素的识别,我们是先裁剪出来,再识别的.我们在裁剪的时候,是以"限DBCI"为参考的
   ,每次裁剪前,都会用模板在图片匹配对应的位置,得到参考坐标,再相对于这个参考坐标裁剪各个元素.
