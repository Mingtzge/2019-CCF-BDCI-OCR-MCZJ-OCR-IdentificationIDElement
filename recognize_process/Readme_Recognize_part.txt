这是识别部分代码的说明文档。
**************************************************参考资料**********************************************************
*识别模型采用CRNN。论文链接：
*							https://arxiv.org/abs/1507.05717
*参考了GitHub上两个模型的TensorFlow实现，链接为：
*											https://github.com/MaybeShewill-CV/CRNN_Tensorflow   
*											https://github.com/bai-shang/crnn_ctc_ocr.Tensorflow
********************************************************************************************************************

文件夹 char_map 保存一个json文件char_map.json，用于将字符映射为对应数字或完成相反操作
文件夹 ckpt_log_save 保存训练结果和记录 
文件夹 config 中包含脚本model_config.py，保存了模型的部分超参数
文件夹 crnn_model 中的脚本crnn_net.py中保存构建CRNN网络的代码
文件夹 data_provider 中包含两个脚本，write_tfrecord.py用于将数据转为tfrecord文件，read_tfrecord.py用于读取tfrecord文件
文件夹 tools中包含脚本train_crnn.py用于训练网络，脚本mytest_crnn.py和test_crnn_jmz.py用于识别图片中的文字
文件夹 test_imgs 中包含待测试图片
文件夹 model_save 中保存的是训练好的模型

此部分需要的环境信息在 Requirement_Recognize_part.txt中。
测试时执行test_crnn_jmz.py，主目录 BDCI_IDCARD中main_process.py调用的示例：
    test_crnn_jmz.recognize_jmz(image_path=args.recognize_image_path, weights_path=args.recognize_weights_path,
              char_dict_path=args.recognize_char_dict_path, txt_file_path=args.recognize_txt_path)
在主目录BDCI_IDCARD下，在终端中调用的示例：
	python ./recognize_process/tools/test_crnn_jmz.py -i ./recognize_process/test_imgs/ -t ./recognize_process/anno_test
	各参数含义为：
		(-i)image_path：     需要识别的图片所在路径     默认是 ./recognize_process/test_imgs/
		(-w)weights_path:    模型权重的路径            默认是 ./recognize_process/model_save/recognize_model
		(-c)char_dict_path:  字典所在路径              默认是 ./recognize_process/char_map/char_map.json
		(-t)txt_file_path:   标注文件所在路径          默认是  ./recognize_process/anno_test/                   
	模型将根据txt_file_path提供的路径，查询该目录下的txt文件，遍历各个txt文件中的图片名，根据image_path提供的图片路径，识别各个图片。

训练时执行train_crnn.py， 在主目录BDCI_IDCARD下，终端调用的示例：
	python ./recognize_process/tools/train_crnn.py -d ./data_tfrecord/ -s ./ckpt_save/
		各参数含义为：
		(-d)dataset_dir：    需要识别的图片所在路径      默认是  None
		(-w)weights_path:    预训练模型权重的路径        默认是  None
		(-c)char_dict_path:  字典所在路径               默认是 ./recognize_process/char_map/char_map.json
		(-s)save_path:       标注文件所在路径           默认是  None                   
	加载路径dataset_dir下的tfrecord用于训练，如果参数weights_path不为None，将加载此预训练模型，训练保存的模型将保存在save_path下。
	训练中的各个超参数，可以在目录model下的model_config.py中修改。一般需要修改的参数有：
		__C.TRAIN.EPOCHS = 580000                          # 训练终止步数
		__C.TRAIN.DISPLAY_STEP = 200                       # 训练过程中可视化步数
		__C.TRAIN.LEARNING_RATE = 30000.0                  # 初始学习率
		__C.TRAIN.BATCH_SIZE = 64                          # batch_size
		__C.TRAIN.LR_DECAY_STEPS = 2000                    # 使用学习率指数衰减，衰减步幅
		__C.TRAIN.LR_DECAY_RATE = 0.94                     # 衰减值
		__C.TRAIN.SAVE_STEPS = 10000                       # 每隔多少步保存一次模型
		

识别模型的训练过程叙述如下：
	1.实验硬件环境：
		CPU：Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz；32核
		内存：128G；
		Swap分区大小：8G；
		GPU： TITAN Xp 显存12G。
	2.实验软件环境：
		详见Requirement_Recognize_part.txt
	3.训练过程：
		（1）.使用开源数据集训练约9万个step，batch size为128， 学习率为0.03， 衰减步幅为10000，衰减值为0.6；数据集链接：
		https://pan.baidu.com/s/1ufYbnZAZ1q0AlK7yZ08cvQ   标注链接： https://pan.baidu.com/s/1jfAKQVjD-SMJSffOwGhh8A，密码：u7bo
		
		（2）.使用开源工具https://github.com/Belval/TextRecognitionDataGenerator，根据字典char_map.json中的字符，生成570万张长度不一（字符个数在4-19之间），类别均匀的数据。
		在（1）的基础上训练，保存学习率为0.02，batch size为128，衰减步幅为10000，衰减值为0.6，训练步数为420000。（至此总共训练了450000步）
		
		（3）.根据身份证背景，模仿生成身份证图片，添加水印并去除后，作为训练数据，生成了140万张训练数据。初始化学习率为0.015，batch size为128，衰减步幅为10000，衰减值为0.6，训练步数为80000。（至此总共训练了530000步）
		
		（4）.使用初赛的1万张训练集和复赛的8000张训练集（拆分出180000张训练图片），在（3）的基础上继续训练，初始化学习率为0.010，batch size为128，衰减步幅为1500，衰减值为0.94，训练步数为90000。（至此总共训练了620000步）


