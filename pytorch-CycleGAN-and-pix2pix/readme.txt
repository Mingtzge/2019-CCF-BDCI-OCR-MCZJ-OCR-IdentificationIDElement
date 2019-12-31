此项目主要引用
**************************************************参考资料**********************************************************
*去水印模型采用gan网络。论文链接：
*                           https://arxiv.org/pdf/1611.07004.pdf
*参考了GitHub上gan pix2pix 项目，链接为：
                            https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix#cyclegan-and-pix2pix-in-pytorch
********************************************************************************************************************
在此项目的基础上,做了些许修改,主要更改的地方:训练的时候加入了数据增强,以及一些结果展示上的修改,但是主体基本上来自于该项目
功能说明:主要用此项目的pix2pix的图片生成方法,去除初赛和复赛中的水印.
模型训练:
数据:数据来自我们自己生成的数据集
参数:在训练和测试的时候,我们对其中的参数进行了调节

模型训练详细步骤:
注:数据集生成方式在数据集生成文件有详细说明


复印无效: 数据生成方式见--->/fake_data_generater/chusai_fuyinwuxiao/readme
数据集1:14万
数据集2:20万
数据集3:30+万
第一轮训练:
时长估计:3~4天
数据dir1:数据集1+数据集2
python train.py --dataroot dir1 --name model_for_example_chusai --model pix2pix --direction AtoB --add_contrast --gan_mode lsgan
                --crop_size 512 --load_size 512 --niter 20 --niter_decay 10 --lr 0.005
命令解释：
具体可在 ./options/base_options.py 和 ./options/train_options.py两个文件中看
--dataroot 数据目录，定位到你解压之后的文件即可，子目录需要包含test val train三个文件夹
--name 实验名 用这个来辨别不同的模型,每次新的训练需要更改一下
第二轮训练:
数据dir2: 数据集3
时长估计14时
python train.py --dataroot dir3 --name model_for_example_chusai --model pix2pix --direction AtoB --add_contrast --gan_mode lsgan
                --crop_size 512 --load_size 512 --niter 0 --niter_decay 3  --lr 0.005 --continue_train
注:此次训练为finetune, 两次的模型名--name 参数应该保持一致


禁止复印:数据生成方式见--->/fake_data_generater/rematch_jinzhifuyin/readme
数据集4:40+万
第一轮训练:
时长估计1~2天
数据dir3:数据集4
python train.py --dataroot dir3 --name model_for_example_fusai --model pix2pix --direction AtoB --add_contrast --gan_mode lsgan
                --crop_size 256 --load_size 256 --niter 2 --niter_decay 5 --lr 0.005

# gan改进去"禁止复印"水印模型实验流程,在第一轮训练的基础上finetune, 模型名--name 参数应该保持一致

1. 全部的训练过程都是使用下面的指令：

   python3 train.py --dataroot dataset --name model_for_example_fusai  --model pix2pix --direction AtoB --checkpoints_dir checkpoint_path --add_contrast --gan_mode lsgan --gpu_ids gpu_id --load_size 256 --crop_size 256 --niter iter_num --niter_decay iter_decay_num --input_nc 1 --output_nc 1 --continue_train --lr learning_rate

   对上面指令中的需要改变的参数进行介绍。

   --dataroot：存放训练数据集的路径，不同的训练阶段需要使用不同的训练集。

   --name：本次训练保存模型的名称，因为整个过程都是在之前的基础上进行微调，所以这个参数与前一阶段训练对应模型的名称是相同的，训练结束后的模型会保存在该命名的文件夹下，并覆盖前一阶段训练的模型。

   --checkpoints_dir: 存放上述模型的文件夹的名称

   --gpu_ids: 指定显卡

   --niter:

   --niter_decay:

   --lr: 此次训练的起始学习率

   下面给出不同阶段训练对应的上述参数。

2. 在815张图片上使用水印平移方法生成4万张图片，水印平移程序第345行传入的第二个参数设置为2，生成的数据分为train、val、test三个部分存放在dir_1文件夹中（dir_1仅做示意用），使用的水印模板是水印平移程序文件夹下的roi_2.jpg，将水印平移程序第332行template_file修改为roi_2.jpg，在之前的模型上进行优化（使用finetune_model名称来示意，该文件夹保存在与训练程序相同层级的model文件夹下）。

   --dataset: dir_1

   --name: finetune_model

   --chechpoints_dir: model

   --niter: 5

   --niter_decay: 5

   --lr: 0.005

3. 在815张图片上使用水印平移方法生成2万余张图片，水印平移程序第345行传入的第二个参数设置为1，生成的数据分为train、val、test三个部分存放在dir_2文件夹中（dir_2仅做示意用），使用的水印模板是水印平移程序文件夹下的roi.jpg，将水印平移程序第332行template_file修改为roi.jpg，在之前的模型上进行优化（使用finetune_model名称来示意，该文件夹保存在与训练程序相同层级的model文件夹下）。

   --dataset: dir_2

   --name: finetune_model

   --chechpoints_dir: model

   --niter: 5

   --niter_decay: 5

   --lr: 0.001

4. 在经过清洗的596张图片上使用水印平移方法生成2万余张图片，水印平移程序第345行传入的第二个参数设置为1，生成的数据分为train、val、test三个部分存放在dir_3文件夹中（dir_3仅做示意用），使用的水印模板是水印平移程序文件夹下的roi.jpg，将水印平移程序第332行template_file修改为roi.jpg，在之前的模型上进行优化（使用finetune_model名称来示意，该文件夹保存在与训练程序相同层级的model文件夹下）。

   --dataset: dir_3

   --name: finetune_model

   --chechpoints_dir: model

   --niter: 2

   --niter_decay: 1

   --lr: 0.001

5. 使用不同的生成水印图片的方法，生成了14万张训练图片，存放在dir_4文件夹下，在之前的模型上进行优化（使用finetune_model名称来示意，该文件夹保存在与训练程序相同层级的model文件夹下）。

   --dataset: dir_4

   --name: finetune_model

   --chechpoints_dir: model

   --niter: 5

   --niter_decay: 5

   --lr: 0.001

6. 从上一阶段使用到的14万张图片中随机选择4万多张图片，使用水印平移程序生成4万多张图片，水印平移程序第345行传入的第二个参数设置为1，生成的数据分为train、val、test三个部分存放在dir_5文件夹中（dir_5仅做示意用），使用的水印模板是水印平移程序文件夹下的roi.jpg，将水印平移程序第332行template_file修改为roi.jpg，对两部分数据进行融合得到了总共9万3千多张训练图片，随val、test存放在dir_5文件夹下，在上一步的基础上进行finetune。

   --dataset: dir_5

   --name: finetune_model

   --chechpoints_dir: model

   --niter: 5

   --niter_decay: 5

   --lr: 0.0001


