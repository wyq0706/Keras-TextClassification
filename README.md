# Keras-TextClassification


# keras_textclassification（代码主体,未完待续...）
    - FastText
    - TextCNN
    - charCNN
    - TextRNN
    - TextRCNN


# run(运行, 以FastText为例)
    - 1. 进入keras_textclassification/m01_FastText目录，
    - 2. 训练: 运行 train.py,   例如: python train.py
    - 3. 预测: 运行 predict.py, 例如: python predict.py
    - 说明: 默认不带pre train的random embedding，训练和验证语料只有100条，完整语料移步下面data查看下载


# keras_textclassification/data
    - 数据下载
      ** github项目中只是上传部分数据，需要的前往链接: https://pan.baidu.com/s/1I3vydhmFEQ9nuPG2fDou8Q 提取码: rket
    - baidu_qa_2019（百度qa问答语料，只取title作为分类样本，17个类，有一个是空''，已经压缩上传）
       - baike_qa_train.csv
       - baike_qa_valid.csv
    - embeddings
       - chinese_L-12_H-768_A-12(取谷歌预训练好点的模型，已经压缩上传)
    - model
       - 预训练模型存放地址

# 项目说明
  - 1. 构建了base基类(网络(graph)、向量嵌入(词、字、句子embedding)),后边的具体模型继承它们，代码简单
  - 2. conf存放项目数据、模型的地址, data存放数据和语料, etl为数据预处理模块,


# 模型与论文paper题与地址
* FastText:   [Bag of Tricks for Efﬁcient Text Classiﬁcation](https://arxiv.org/abs/1607.01759)
* TextCNN：   [ConvolutionalNeuralNetworksforSentenceClassiﬁcation](https://arxiv.org/abs/1408.5882)
* charCNN：   [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615)
* TextRNN：   [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)
* RCNN：      [Recurrent Convolutional Neural Networks for Text Classification](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf)
* DCNN:       [A Convolutional Neural Network for Modelling Sentences](https://arxiv.org/abs/1404.2188)


# 参考/感谢
* 文本分类项目:   [https://github.com/mosu027/TextClassification](https://github.com/mosu027/TextClassification)
* 文本分类看山杯: [https://github.com/brightmart/text_classification](https://github.com/brightmart/text_classification)
* Kashgari项目: [https://github.com/BrikerMan/Kashgari](https://github.com/BrikerMan/Kashgari)
* 文本分类Ipty : [https://github.com/lpty/classifier](https://github.com/lpty/classifier)
* keras文本分类: [https://github.com/ShawnyXiao/TextClassification-Keras](https://github.com/ShawnyXiao/TextClassification-Keras)