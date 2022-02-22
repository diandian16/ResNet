#!/usr/bin/env python
# coding: utf-8

# # 一、项目背景介绍
# 
# 对男生来说，区分女生化妆品很累吧，瓶瓶盘盘的有一堆的，利用迁移学习的方法来训练这些数据。
# 
# 随着越来越多的机器学习应用场景的出现，而现有表现比较好的监督学习需要大量的标注数据，标注数据是一项枯燥无味且花费巨大的任务，所以迁移学习受到越来越多的关注。
# 
# 迁移学习是一种机器学习方法，就是把为任务 A 开发的模型作为初始点，重新使用在为任务 B 开发模型的过程中。
# 
# 

# # 二、数据介绍
# 
# 化妆品分类  https://aistudio.baidu.com/aistudio/datasetdetail/120534 
# 
# 八类化妆品分别是 ['bbCream', 'blush', 'eyebrow', 'eyeshadow', 'foundation', 'lipstick', 'mascara', 'nail_polish']
# 
# 有训练数据集和测试数据集两类

# # 三、模型介绍
# 
# 整体项目构建如下，先从数据准备开始，使用到模型时候会介绍的

# ![](https://ai-studio-static-online.cdn.bcebos.com/6734476c8d4a426a991c6bcef20dca4126e2dcbae7d847bead6168c07768a900)
# 
# 

# In[ ]:


# 引入os库,方便路径操作
import os
# 引入文件操作模块
import shutil
# 引入百度paddle模块
import paddle as paddle
# 引入百度飞桨的fluid模块,方便
import paddle.fluid as fluid
# 方便设置参数
from paddle.fluid.param_attr import ParamAttr
# 引入自行封装的reader文件
import work.keen_reader
# 引入numpy库,方便计算和保存数据
import numpy as np
# 引入pandas库,方便使用
import pandas as pd
# 引入随机数
import random
# 引入日志库,方便记录操作的结果
import logging 


# In[ ]:



"""注意！！！

该命令只需运行一次

该行命令用于解压缩
"""

# 获取预训练模型
get_ipython().system('wget http://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar')
get_ipython().system('tar -xvf ResNet101_pretrained.tar')
#解压数据集图片
get_ipython().system('unzip data/data40449/D0001.zip ')


# ### 配置超参

# In[ ]:


# tips:在这里配置主要修改的参数,在下面就用变量的方式来调用这些超参即可

train_core1 = { 
    # 输入size大小,建议保持一致
    "input_size": [3, 224, 224], 
    # 使用这个项目是要把目标图像分成多少种类
    "class_dim":8,  # 分类数,
    # 主要修改的超参学习率,可以试试0.001,0.005,0.01,0.02之类的，调节正确率波动
    "learning_rate":0.002,
    # 建议使用GPU,否则训练时间会很长
    "use_gpu": True,
    # 前期的训练轮数,你可以尝试着去增加试试看
    "num_epochs": 30, #训练轮数
    # 当达到想要的准确率就立刻保存下来当时的模型
    "last_acc":0.4
} 


# ### 初始化日志文件
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/4b657823d8e0476a9b1a202d4dc98665a8dd29c623db409dbd21e6085d2b092f)
# 

# In[ ]:


# 设置一个全局的日志变量
global logger 
logger = logging.getLogger() 
logger.setLevel(logging.INFO) 
log_path = os.path.join(os.getcwd(), 'logs') #当前目录下创建log文件夹
if not os.path.exists(log_path): 
    os.makedirs(log_path) 
log_name = os.path.join(log_path, 'train.log') #日志文件的名字
# sh = logging.StreamHandler() 
fh = logging.FileHandler(log_name, mode='w') 
fh.setLevel(logging.DEBUG) 
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s") 
fh.setFormatter(formatter) 
# sh.setFormatter(formatter) 
# logger.addHandler(sh) 
logger.addHandler(fh) 
# 记录此次运行的超参,方便日后做记录进行比对
logger.info(train_core1)

# tips:小伙伴们可以在这里初始化日志的时候加上日期,更为方便.


# ### 数据采集与预处理
# 
# 本次研究的对象为化妆品，使用的数据来源于百度官方提供的化妆品-8种分类数据集包含有8种不同化妆品的图片
# 
# 
# 
# 应用留出法，随机将90%的样本设置为训练集，10%的样本设置为测试集。
# 
# 
# 

# In[ ]:


#按比例随机切割数据集
train_ratio=0.90

#tips: 小伙伴们可以在这里修改训练集和测试集的比例,比如0.5,0.7,0.9,比例越高效果相对来说较好

train=open('train_split_list.txt','w')
val=open('val_split_list.txt','w')
with open('train_list.txt','r') as f:
    #with open('data_sets/cat_12/train_split_list.txt','w+') as train:
    lines=f.readlines()
    for line in lines:
        line=line.replace("\\","/")
        line=line.replace(" ","\t")
        train.write(line)
with open('val_list.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        line=line.replace("\\","/")
        line=line.replace(" ","\t")
        val.write(line)

train.close()
val.close()
                


# 为了增加训练集的数据量，提高模型的泛化能力，对训练集进行数据增强处理
# 
# 应用数据增强技术，对已有图片做缩放、随机旋转、随机裁剪、对比度调整、色调调整以及饱和度调整，数据增强后，大幅提升了训练样本数量。
# 
# 为了之后的使用方便,进行了封装
# 
# 对输入的图片进行归一化，保证输入的信息类型一致。
# 

# In[ ]:


# tips:因为我已经封装成为文件了,你可以仔细读读work/keen_reader.py文件的

##获取数据（batch_size可以调一次多少张）
train_reader = paddle.batch(work.keen_reader.train(), batch_size=30)
test_reader = paddle.batch(work.keen_reader.val(), batch_size=32)
logger.info("成功加载数据") 




# tips:解除注释,可以查看一下获得的数据是什么样子的
# sampledata=next(train_reader())
# print(sampledata)


# ### 迁移学习介绍
# 
# 在本次研究中，为保证ResNet101神经网络模型的泛化能力，采取迁移学习策略。
# 
# 采用两步化方案：第一步是预训练过程，首先，为阻止全连接层进行反向传播，去除最后一层全连接层,冻结整个整个卷积神经网络，
# 返回进行卷积后的结果再依赖其构建一个分类为12层的全连接层，卷积神经网络载入官方的预训练模型。
# 接着，使用AdamOptimizer优化器以较小的学习率与训练数据规模，对训练集进行尝试性的训练。在完成预训练后，保存新的分类器参数。
# 
# [百度飞桨模型库](https://www.paddlepaddle.org.cn/modelbasedetail/resnet)
# 
# [PaddlePaddle预训练模型](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/README.md)
# 
# (http://www.clzly.xyz/2020/python/bf62f4ed/%E5%9F%BA%E4%BA%8E%E7%BD%91%E7%BB%9C%E7%9A%84%E6%B7%B1%E5%BA%A6%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E7%9A%84%E7%A4%BA%E6%84%8F%E5%9B%BE.jpg)
# 
# 第二步，加载新分类器参数，开放全连接层上层的卷积神经网络部分，允许训练过程中从全连接层到网络浅层的反向传播。
# 接着，使用SGD优化器以较小的学习率和较大的训练数据规模对模型进行最后的调试。

# ### 定义残差网络
# 
# [论文](https://arxiv.org/abs/1512.03385)
# 
# [ResNet论文翻译——中文版](http://noahsnail.com/2017/07/31/2017-07-31-ResNet%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
# 
# 传统的卷积层或全连接层在传递信息时，或多或少会存在信息丢失、损耗等问题。ResNet在某种程度上解决了这个问题，通过直接将输入信息绕道传到输出，保护信息的完整性，整个网络只需要学习输入、输出差别的那一部分，简化学习目标和难度。
# 
# 作者将bottleneck拆分成多个分支，提出了神经网络中的第三个维度（另外两个维度分别为depth，神经网络层数深度，width，宽度，channel数），命名为`Cardinality`，并在多个数据集中证明了将bottleneck拆分能够降低训练错误率和提高准确率。
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/3b47cf39969547b4bfd67eb6a1303dd8bc42b9e301a346b1ab7171ab2b9e8df0)
# 
# 
# 再看一下VGG19，一个34层深的普通卷积网络和34层深的ResNet网络的对比图。可以看到普通直连的卷积网络和ResNet的最大区别在于，ResNet有很多旁路的支线将输入直接连到后面的层，使得后面的层可以直接学习残差，这种结构也被称为shortcut或skip connections。
# 
# <img src="http://noahsnail.com/images/resnet/Figure_3.jpeg" width = "100%" align="center" />
# 
# 
# 图像分类模型架构，详见[百度飞桨官方的Github
# ResNet101](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/image_classification)

# In[ ]:


def resnet(input):
    def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1, act=None, name=None):
        conv = fluid.layers.conv2d(input=input,
                                   num_filters=num_filters,
                                   filter_size=filter_size,
                                   stride=stride,
                                   padding=(filter_size - 1) // 2,
                                   groups=groups,
                                   act=None,
                                   param_attr=ParamAttr(name=name + "_weights"),
                                   bias_attr=False,
                                   name=name + '.conv2d.output.1')
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(input=conv,
                                       act=act,
                                       name=bn_name + '.output.1',
                                       param_attr=ParamAttr(name=bn_name + '_scale'),
                                       bias_attr=ParamAttr(bn_name + '_offset'),
                                       moving_mean_name=bn_name + '_mean',
                                       moving_variance_name=bn_name + '_variance', )

    def shortcut(input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input
            
    def bottleneck_block(input, num_filters, stride, name):
        conv0 = conv_bn_layer(input=input,
                              num_filters=num_filters,
                              filter_size=1,
                              act='relu',
                              name=name + "_branch2a")
        conv1 = conv_bn_layer(input=conv0,
                              num_filters=num_filters,
                              filter_size=3,
                              stride=stride,
                              act='relu',
                              name=name + "_branch2b")
        conv2 = conv_bn_layer(input=conv1,
                              num_filters=num_filters * 4,
                              filter_size=1,
                              act=None,
                              name=name + "_branch2c")

        short = shortcut(input, num_filters * 4, stride, name=name + "_branch1")

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu', name=name + ".add.output.5")

    depth = [3, 4, 23, 3]
    num_filters = [64, 128, 256, 512]

    conv = conv_bn_layer(input=input, num_filters=64, filter_size=7, stride=2, act='relu', name="conv1")
    conv = fluid.layers.pool2d(input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            if block == 2:
                if i == 0:
                    conv_name="res"+str(block+2)+"a"
                else:
                    conv_name="res"+str(block+2)+"b"+str(i)
            else:
                    conv_name="res"+str(block+2)+chr(97+i)
            conv = bottleneck_block(input=conv,
                                    num_filters=num_filters[block],
                                    stride=2 if i == 0 and block != 0 else 1,
                                    name=conv_name)

    pool = fluid.layers.pool2d(input=conv, pool_size=7, pool_type='avg', global_pooling=True)
    return pool


# ### 定义输入层、主程序、损失函数和准确率函数、优化函数

# In[ ]:


##定义输入层
image=fluid.layers.data(name='image',shape=train_core1["input_size"],dtype='float32')
label=fluid.layers.data(name='label',shape=[1],dtype='int64')


##停止梯度下降（冻结网络层stop）
pool=resnet(image)
pool.stop_gradient=True

##创建主程序来预训练
base_model_program=fluid.default_main_program().clone()
model=fluid.layers.fc(input=pool,size=train_core1["class_dim"],act='softmax')


##定义损失函数和准确率函数
cost=fluid.layers.cross_entropy(input=model,label=label)
avg_cost=fluid.layers.mean(cost)
acc=fluid.layers.accuracy(input=model,label=label)


##定义优化方法
optimizer=fluid.optimizer.AdamOptimizer(learning_rate=train_core1["learning_rate"])
opts=optimizer.minimize(avg_cost)

# tips:优化方法并不是只有Adam的,还有许多其他的方法,各有优劣,建议多加尝试

##定义训练场所
use_gpu=train_core1["use_gpu"]
place=fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe=fluid.Executor(place)

##进行参数初始化
exe.run(fluid.default_startup_program())
logger.info("成功参数初始化") 


# # 四、模型训练

# ### 加载预训练模型

# In[ ]:


#每次重启后运行一次
##预训练模型路径
src_pretrain_model_path='ResNet101_pretrained'
logger.info("开始加载预训练模型") 
##判断模型文件是否存在
def if_exit(var):
    path=os.path.join(src_pretrain_model_path,var.name)
    exist=os.path.exists(path)
    if exist:
      # print('Load model: %s' % path)
      return exist

##加载模型文件，且只加载存在模型的模型文件
fluid.io.load_vars(executor=exe,dirname=src_pretrain_model_path,predicate=if_exit,main_program=base_model_program)
logger.info("加载预训练模型成功") 


# ### 定义数据维度并开始训练

# In[ ]:


##定义数据维度
feeder=fluid.DataFeeder(place=place,feed_list=[image,label])
save_pretrain_model_path='models/step-1_model/'
# 预期的准确率
last_acc=train_core1["last_acc"]
# 初始的准确率
now_acc=0
logger.info("开始第一批训练数据。。。") 
for pass_id in range(train_core1["num_epochs"]):
    for batch_id,data in enumerate(train_reader()):
        train_cost,train_acc=exe.run(program = fluid.default_main_program(),feed=feeder.feed(data),fetch_list=[avg_cost,acc])
        if batch_id%50==0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))
    
    now_acc=train_acc
    if now_acc>last_acc and now_acc!=1:
        last_acc=now_acc
        logger.info("临时保存第{0}批次的训练结果，准确率为{1}".format(pass_id,now_acc)) 
        ##删除旧的模型文件
        shutil.rmtree(save_pretrain_model_path,ignore_errors=True)
        ##创建保存模型文件记录
        os.makedirs(save_pretrain_model_path)
        ##保存参数模型
        fluid.io.save_params(executor=exe,dirname=save_pretrain_model_path)
logger.info("第一批训练数据结束。") 





# #  **Step-2**
# 
# 此处需要重启环境，否则会报错

# In[ ]:


# 此处同上
import os
import shutil
import paddle as paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
import work.keen_reader
import logging


# ### 定义超参

# In[ ]:


train_core2 = { 
    "input_size": [3, 224, 224], 
    "class_dim": 8,  # 分类数,
    # 定义学习率,这里也是可以多加修改的
    "learning_rate":0.002,
    # 定义sgd的学习率
    "sgd_learning_rate":0.0002,
    # 学习率自调整
    "lrepochs":[20,40,60,80,100],
    "lrdecay":[1,0.5,0.25,0.1,0.01,0.002],
    "use_gpu": True,
    # 这里的训练轮数自我把握,因为在后期可能训练500轮在最后的200轮已经没有变化了
    "num_epochs": 30, #训练轮数
    # 想要达到的最终准确率
    "last_acc":0.7

} 


# ### 初始化日志文件

# In[ ]:


#初始化日志文件

global logger 
logger = logging.getLogger() 
logger.setLevel(logging.INFO) 
log_path = os.path.join(os.getcwd(), 'logs') #当前目录下创建log文件夹
if not os.path.exists(log_path): 
    os.makedirs(log_path) 
log_name = os.path.join(log_path, 'test.log') #日志文件的名字
# sh = logging.StreamHandler() 
fh = logging.FileHandler(log_name, mode='w') 
fh.setLevel(logging.DEBUG) 
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s") 
fh.setFormatter(formatter) 
# sh.setFormatter(formatter) 
# logger.addHandler(sh) 
logger.addHandler(fh) 

logger.info("train_core2 config: %s", str(train_core2)) 


# ### 加载数据信息

# In[ ]:


##获取数据
train_reader = paddle.batch(work.keen_reader.train(), batch_size=30)
test_reader = paddle.batch(work.keen_reader.val(), batch_size=32)
logger.info("成功加载数据") 


# ### 定义新的残差网络

# In[ ]:


#tips: 新定义resnet，加上fc
def resnet(input,class_dim):
    def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1, act=None, name=None):
        conv = fluid.layers.conv2d(input=input,
                                   num_filters=num_filters,
                                   filter_size=filter_size,
                                   stride=stride,
                                   padding=(filter_size - 1) // 2,
                                   groups=groups,
                                   act=None,
                                   param_attr=ParamAttr(name=name + "_weights"),
                                   bias_attr=False,
                                   name=name + '.conv2d.output.1')
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(input=conv,
                                       act=act,
                                       name=bn_name + '.output.1',
                                       param_attr=ParamAttr(name=bn_name + '_scale'),
                                       bias_attr=ParamAttr(bn_name + '_offset'),
                                       moving_mean_name=bn_name + '_mean',
                                       moving_variance_name=bn_name + '_variance', )

    def shortcut(input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input
            
    def bottleneck_block(input, num_filters, stride, name):
        conv0 = conv_bn_layer(input=input,
                              num_filters=num_filters,
                              filter_size=1,
                              act='relu',
                              name=name + "_branch2a")
        conv1 = conv_bn_layer(input=conv0,
                              num_filters=num_filters,
                              filter_size=3,
                              stride=stride,
                              act='relu',
                              name=name + "_branch2b")
        conv2 = conv_bn_layer(input=conv1,
                              num_filters=num_filters * 4,
                              filter_size=1,
                              act=None,
                              name=name + "_branch2c")

        short = shortcut(input, num_filters * 4, stride, name=name + "_branch1")

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu', name=name + ".add.output.5")

    depth = [3, 4, 23, 3]
    num_filters = [64, 128, 256, 512]

    conv = conv_bn_layer(input=input, num_filters=64, filter_size=7, stride=2, act='relu', name="conv1")
    conv = fluid.layers.pool2d(input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            if block == 2:
                if i == 0:
                    conv_name="res"+str(block+2)+"a"
                else:
                    conv_name="res"+str(block+2)+"b"+str(i)
            else:
                    conv_name="res"+str(block+2)+chr(97+i)
            conv = bottleneck_block(input=conv,
                                    num_filters=num_filters[block],
                                    stride=2 if i == 0 and block != 0 else 1,
                                    name=conv_name)

    pool = fluid.layers.pool2d(input=conv, pool_size=7, pool_type='avg', global_pooling=True)
    output=fluid.layers.fc(input=pool,size=class_dim,act='softmax')#全连接层不冻结，全参与
    return output


# ### 定义输入层、主程序、损失函数和准确率函数、优化函数

# In[ ]:


##定义输入层
image = fluid.layers.data(name='image', shape=train_core2["input_size"], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
logger.info("成功定义输入层") 
##获取分类器
model = resnet(image,train_core2["class_dim"])

##获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

##获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

##定义优化方法
optimizer=fluid.optimizer.SGD(learning_rate=train_core2["sgd_learning_rate"])
opts=optimizer.minimize(avg_cost)

# tips:不同的优化方法都可以多加尝试,尤其是第一步和第二步优化方法相同或者不同都会有"惊""喜"哦!

##定义一个使用GPU的执行器
use_gpu=train_core2["use_gpu"]
place=fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe=fluid.Executor(place)

##进行参数初始化
exe.run(fluid.default_startup_program())
logger.info("成功进行参数初始化") 


# ### 加载经过处理的模型

# In[ ]:


##经过step-1处理后的的预训练模型
pretrained_model_path = 'models/step-1_model/'

##加载经过处理的模型
fluid.io.load_params(executor=exe, dirname=pretrained_model_path)
last_acc=train_core2["last_acc"]
logger.info("成功加载第一步的预训练模型！") 


# In[ ]:


from visualdl import LogWriter


# ### 定义数据维度并开始训练

# In[24]:


##定义数据维度
feeder=fluid.DataFeeder(place=place,feed_list=[image,label])
#value = [i/1000.0 for i in range(1000)]
now_acc=0
logger.info("开始第二批训练数据。。。") 
##保存预测模型
save_path = 'models/step_2_model/'
with LogWriter(logdir="./log/scalar_test/train") as writer:
    for pass_id in range(train_core2["num_epochs"]):
        ##训练
        for batch_id,data in enumerate(train_reader()):
            train_cost,train_acc=exe.run(program=fluid.default_main_program(),feed=feeder.feed(data),fetch_list=[avg_cost,acc])
            if batch_id%50==0:
                print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                    (pass_id, batch_id, train_cost[0], train_acc[0]))
        ##测试
        test_accs=[]
        test_costs=[]
        for batch_id,data in enumerate(test_reader()):
            test_cost,test_acc=exe.run(program=test_program,feed=feeder.feed(data), fetch_list=[avg_cost,acc])
            test_accs.append(test_acc[0])
            test_costs.append(test_cost[0])
        test_cost = (sum(test_costs) / len(test_costs))
        test_acc = (sum(test_accs) / len(test_accs))
        logger.info('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))
        writer.add_scalar(tag="acc", step=pass_id, value=test_acc)
        writer.add_scalar(tag="loss", step=pass_id, value=test_cost)
        now_acc=test_acc
        
        if now_acc>last_acc:
            last_acc=now_acc
            logger.info("临时保存第 {0}批次的训练结果，准确率为 acc1 {1}".format(pass_id, now_acc))
            
            ##删除旧的模型文件
            shutil.rmtree(save_path, ignore_errors=True)
            ##创建保持模型文件目录
            os.makedirs(save_path)
            ##保存预测模型
            fluid.io.save_inference_model(save_path, feeded_var_names=[image.name], target_vars=[model], executor=exe)
    logger.info("第二批训练数据结束。") 






# # 五、模型评估

# ### 可视化acc结果和loss结果
# 
# 目前经典版的aistudio没有VisualDL的可视化了，所以是新建了一个BMLCodeLab里面进行可视化的
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/f86a3c2b642a4377960dfd474ebda5f6fa225c177f3f41ea9c47f53172d3db43)
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/f99eee077ffa43cdb0d5ea79ad6904562992558bd7b04a0880688bf7cd4aa6e3)
# 
# 
# 

# ![](https://ai-studio-static-online.cdn.bcebos.com/d79074fe74044ac0bd4c33118236ddb46c67f2ec649348c0801fd7e2114676fc)
# 

# ### Setp3 生成预测结果文件

# In[ ]:



test_list_imagepath=[]
test_list_labels=[]
test_dict={}

with open("./test_list.txt") as f1:
    for line1 in f1:
        line1=line1.replace('\\','@')
        line1=line1.replace('@','/')
        test_list_imagepath.append(line1[:-3])
        test_list_labels.append(line1[-2])
        test_dict[line1[:-3]]=line1[-2]


# In[ ]:



print(test_list_labels)


# In[ ]:



print(test_list_imagepath)


# In[ ]:



SAVE_DIRNAME = './models/step_2_model'  # 保存好的 inference model 的路径
abs_path = r'./' # 测试文件夹的真实路径
#########################################################################
# coding:utf-8
#from __future__ import print_function
import os
import json

import paddle
import paddle.fluid as fluid
import numpy as np
from PIL import Image
import sys

TOP_K = 1
DATA_DIM = 224

use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model( SAVE_DIRNAME, exe
                # model_filename='model',
                # params_filename='params'
                # model_filename = 'fc_0.w_0',
                # params_filename = 'params'
                )


def real_infer_one_img(im):
    infer_result = exe.run(
        inference_program,
        feed={feed_target_names[0]: im},
        fetch_list=fetch_targets)

    # print(infer_result)
    # 打印预测结果
    mini_batch_result = np.argsort(infer_result)  # 找出可能性最大的列标，升序排列
    # print(mini_batch_result.shape)
    mini_batch_result = mini_batch_result[0][:, -TOP_K:]  # 把这些列标拿出来
    mini_batch_result = mini_batch_result.flatten() #拉平了，只吐出一个 array
    mini_batch_result = mini_batch_result[::-1] #逆序
    return mini_batch_result


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

def process_image(img_path):
    img = Image.open(img_path)
    img = resize_short(img, target_size=256)
    img = crop_image(img, target_size=DATA_DIM, center=True)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img).astype(np.float32).transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std

    img = np.expand_dims(img, axis=0)
    return img


def convert_list(my_list):
    my_list = list(my_list)
    my_list = map(lambda x:str(x), my_list)
    # print('_'.join(my_list))
    return '_'.join(my_list)


def infer(file_path):
    im = process_image(file_path)
    result = real_infer_one_img(im)
    result = convert_list(result)
    return result




def createCSVFile(cat_12_test_path,test_list_imagepath):
    lines = []

    # 获取所有的文件名
    
    for file_name in test_list_imagepath:
        file_name = file_name
        file_abs_path = os.path.join(cat_12_test_path, file_name)
        result_classes = infer(file_abs_path)

        file_predict_classes = result_classes

        line = '%s,%s\n'%(file_name, file_predict_classes)
        lines.append(line)

    with open('result.csv', 'w') as f:
        f.writelines(lines)



createCSVFile(abs_path,test_list_imagepath)
print("成功输出结果文件")


# ### models/step_2_model的结果与正确label的对比
# 
# result的部分结果![](https://ai-studio-static-online.cdn.bcebos.com/519d55838593428e9f38a7ee66119c7e10380cd0428b42308239b731738841b8)，正确test_list.label![](https://ai-studio-static-online.cdn.bcebos.com/973eee1d8fc548d49070748db073ca5a1b59debd6b094c78a1fd627ae8e5e4be)
# 
# 大概是22个里面错了三个，其中foundation错误率比较高一点
# 
# 

# # 六、个人小结
# 
# 经过几次测试，方法二的正确率远远高于方法一，炼丹之路还很长
# 
# 通过学习这个项目,我大概可以更进一步的掌握ResNet的网络的基础结构和迁移学习的应用
# 
# 

# # 七、参考文献
# 
# 猫十二分类-飞桨图像分类帮我云撸猫（基于ResNet的图像分类）[https://aistudio.baidu.com/aistudio/projectdetail/474305?channel=0&channelType=0&shared=1](https://aistudio.baidu.com/aistudio/projectdetail/474305?channel=0&channelType=0&shared=1)
