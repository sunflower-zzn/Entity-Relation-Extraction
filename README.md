# Overview

本项目是一个简单的知识图谱构建实现，采用流水线思路：先对文本进行实体识别，然后将识别出的实体和对应的文本进行关系抽取，主要基于以下模型&技术栈：

- 命名实体识别（NER）：[百度LAC模型](https://github.com/baidu/lac)
- 关系抽取：[基于BERT实现关系抽取](https://github.com/Ricardokevins/Bert-In-Relation-Extraction)

# Environment

```shell
# conda环境生成
conda create --name bert python=3.6
# pytorch（阿里云学生机只支持cpu……）
conda install pytorch==1.5.1 torchvision==0.6.1 cpuonly -c pytorch
# transformers（使用国内源）
pip install transformers==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
# flask（python web server）
conda install flask
conda install flask-cors
# 百度LAC开源项目（国内源）
pip install LAC -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# Run

```shell
# train model
python loader.py  # Preprocess the downloaded data
python train.py   # train bert fine-tune

# start web-server ( port:5590 )
kill -9 $(lsof -i:5590 -t) # If the port is occupied
nohup python main.py &
```

# Structure

- **bert-base-chinese/**
  - bert中文预训练模型，下载地址：https://huggingface.co/bert-base-chinese#
- **loader.py**
  - 数据：[Baidu Research Open-Access Dataset - Download](https://ai.baidu.com/broad/download?dataset=dureader) 选择**Knowledge Extraction**下载 train_data.json 和 dev_data.json 
  - 预处理数据，生成 train.json 和 dev.json 
  - 提供加载模型的loader函数供其他文件调用
- **model.py**
  - 定义BERT_Classifier类
- **train.py**
  - 根据bert预训练模型进行fine-tune，并输出net模型（文件名表示正确率）
- **demo.py**（需要训练后的net模型）
  - **caculate_acc**：计算每一个类别的正确率
  - **demo_output**：随机选择样本，输出原文，实体对以及预测的关系，即实例输出
- **main.py**（需要训练后的net模型）
  - Python-Server 入口，启动一个python服务，可以通过url访问（接口文档见下一节）
  - **注**：由于作者部署的云服务器不支持cuda，所以改用了torch+cpu，您可以参考demo.py中的demo_output函数来使用GPU加速

# Model

训练数据说明：

- 训练集：36w条训练数据
- 测试集：45577条测试数据

训练参数：

- 10 Epoch，0.001学习率，设置label共有49种（包含UNK，代表新关系和不存在关系）

训练结果：

- fine-tune后模型在测试集上正确率达到95%

# Interface

## 1. Identity entity

**URL**

localhost:5590/identity

**RequestBody**

```
{
    "text": 文本（最好不要太长，可以测试一下极限情况）,
    "is_ignore": 是否忽略实体个数少于两个的分句
}
```

**Response**

段落拆分为数个句子，对于每一个句子返回text和entity列表，目前支持类型如下：

- 人名
- 地名
- 机构名
- 时间
- 专有名词
- 作品名

```
{
    "errmsg": null,
    "context": [
        {
            "text": "《初中物理竞赛热点专题/竞赛热点专题丛书》是2001年湖南师范大学出版社出版的图书，作者是武建谋，宋善炎，严定新",
            "entity": [
                {
                    "ent": "初中物理竞赛热点专题/竞赛热点专题丛书",
                    "type": "作品名"
                },
                {
                    "ent": "2001年",
                    "type": "时间"
                },
                {
                    "ent": "湖南师范大学出版社",
                    "type": "组织名"
                },
                {
                    "ent": "武建谋",
                    "type": "人名"
                },
                {
                    "ent": "宋善炎",
                    "type": "人名"
                },
                {
                    "ent": "严定新",
                    "type": "人名"
                }
            ]
        }
    ]
}
```

## 2. Relation Extraction

**URL**

localhost:5590/relation/extraction

**RequestBody**

每一个句子支持输入超过两个entity（以列表形式），后端会自动两两组合进行关系抽取，但最好不要超过5个实体，因为服务器太烂了会非常慢！

```
[
    {
        "text": 句子,
        "entitys": [
            ent1,
            ent2,
            ent3
        ]
    },
    {
        "text": 句子,
        "entitys": [
            ent1,
            ent2,
            ent3
        ]
    }
]
```

**Response**

```
{
    "errmsg": null,
    "context": [
        {
            "text": 句子
            "relations": [
                {
                    "ent1": "周星驰",
                    "ent2": "喜剧之王",
                    "rel": "主演"
                },
                {
                    "ent1": "周星驰",
                    "ent2": "独门秘笈",
                    "rel": "导演"
                },
                {
                    "ent1": "喜剧之王",
                    "ent2": "独门秘笈",
                    "rel": "改编自"
                }
            ]
        },
        {
            "text": 句子,
            "relations": [
                {
                    "ent1": "周星驰",
                    "ent2": "喜剧之王",
                    "rel": "主演"
                }
            ]
        }
    ]
}
```

## Test text

**text1**：如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈。爱德华·尼科·埃尔南迪斯（1986-），是一位身高只有70公分哥伦比亚男子，体重10公斤，只比随身行李高一些，2010年获吉尼斯世界纪录正式认证，成为全球当今最矮的成年男人。《身外身梦中梦》是连载于晋江文学城的一部原创类小说，作者是苍生笑。

**text2**：《娘家的故事第二部》是张玲执导，林在培、何赛飞等主演的电视剧。禅意歌者刘珂矣《一袖云》中诉知己…绵柔纯净的女声，将心中的万水千山尽意勾勒于这清素画音中。歌手银临毕业于南京大学。外交部华春莹毕业于南京大学。
