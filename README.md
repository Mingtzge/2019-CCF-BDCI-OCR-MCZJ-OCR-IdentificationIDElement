项目整体文件结构说明:(按照处理流程介绍,具体文件介绍见文件内的readme)

1.cut_twist_process
    剪切、翻转部分代码；用于将身份证正反面从原始图片中切分出来

2.watermask_remover_and_split_data
    进行水印去除,身份证切割,提取文字部分,滤波

3.去水印模型项目pytorch-CycleGAN-and-pix2pix
**************************************************参考资料**********************************************************
*去水印模型采用gan网络。论文链接：
*                           https://arxiv.org/pdf/1611.07004.pdf
*参考了GitHub上gan pix2pix 项目，链接为：
                            https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix#cyclegan-and-pix2pix-in-pytorch
********************************************************************************************************************

4.文字识别模块recognize_process
用于识别图片中的文字信息
**************************************************参考资料**********************************************************
*识别模型采用CRNN。论文链接：
*                           https://arxiv.org/abs/1507.05717
*参考了GitHub上两个模型的TensorFlow实现，链接为：
*                                           https://github.com/MaybeShewill-CV/CRNN_Tensorflow
*                                           https://github.com/bai-shang/crnn_ctc_ocr.Tensorflow
********************************************************************************************************************

5.data_correction_and_generate_csv_file
对识别结果进行纠正,以及生成最终的csv文件

6.data_temp
中间数据存放目录,目录名称:实验名+日期

7.ckpt_save
文字识别模型存放目录

8.CCFTestResultFixValidData_release.csv
生成的结果文件

9.main_process.py
执行脚本

10.Requirement.txt
特殊的环境要求

完整执行示例:
CPU执行,单进程:
python main_process.py --test_experiment_name test_example --test_data_dir /home/ubuntu/test_for_final/ --gan_ids -1 --pool_num 0
参数详解:
--test_experiment_name:实验名,将决定中间数据结果存放目录
--test_data_dir: 中间结果存放目录
--gan_ids: 去水印模型:如果是-1 则是cpu运行, 大于0,则是GPU
--pool_num 0单进程 大于0多进程
其他参数参考main_process.py中的help


!!!注:复现的测试数据跟初赛和复赛的数据格式需要保持一致,每面身份证左上角需要有:"仅限DBCI比赛(复赛)使用"字样,
      且字体大小格式位置应该跟初赛和复赛的保持一致,否则将严重影响识别的准确性甚至代码运行出错
   原因:对于身份证各个元素的识别,我们是先裁剪出来,再识别的.我们在裁剪的时候,是以"仅限DBCI比赛(复赛)使用"为参考的
   ,每次裁剪前,都会用模板在图片匹配对应的位置,得到参考坐标,再相对于这个参考坐标裁剪各个元素.