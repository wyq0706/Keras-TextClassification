# coding=utf-8
import datetime
import pathlib
import sys
import os

from pandas import np

project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(project_path)

# 地址
from keras_textclassification.conf.path_config import path_model, path_fineture, path_model_dir, path_hyper_parameters
# 数据预处理, 删除文件目录下文件
from keras_textclassification.data_preprocess.text_preprocess import PreprocessTextMulti, delete_file, load_json
# 模型图
from keras_textclassification.m02_TextCNN.graph import TextCNNGraph as Graph
# 计算时间
import time


def append_log(hyper_parameters, acc, not_none_acc, threshold):
    f = open('log.txt', 'a', encoding='utf-8')
    wt = [str(datetime.datetime.now()), str(hyper_parameters), "\nacc: " + str(acc),
          "\nnot none acc: " + str(not_none_acc), "\nthreshold: " + str(threshold), "\n"]
    f.writelines(wt)
    f.close()


def train(hyper_parameters=None, rate=1.0):
    if not hyper_parameters:
        hyper_parameters = {
            'len_max': 50,  # 句子最大长度, 固定推荐20-50, bert越长会越慢, 占用空间也会变大, 本地win10-4G设为20就好, 过大小心OOM
            'embed_size': 200,  # 字/词向量维度, bert取768, word取300, char可以更小些
            'vocab_size': 21128,  # 这里随便填的，会根据代码里修改
            'trainable': True,  # embedding是静态的还是动态的, 即控制可不可以微调
            'level_type': 'char',  # 级别, 最小单元, 字/词, 填 'char' or 'word', 注意:word2vec模式下训练语料要首先切好
            'embedding_type': 'random',  # 级别, 嵌入类型, 还可以填'xlnet'、'random'、 'bert'、 'albert' or 'word2vec"
            'gpu_memory_fraction': 0.66,  # gpu使用率
            'model': {'label': 21,  # 类别数
                      'batch_size': 100,  # 批处理尺寸, 感觉原则上越大越好,尤其是样本不均衡的时候, batch_size设置影响比较大
                      'dropout': 0.5,  # 随机失活, 概率
                      'decay_step': 100,  # 学习率衰减step, 每N个step衰减一次
                      'decay_rate': 0.9,  # 学习率衰减系数, 乘法
                      'epochs': 30,  # 训练最大轮次
                      'patience': 3,  # 早停,2-3就好
                      'lr': 1e-3,  # 学习率, bert取5e-5, 其他取1e-3, 对训练会有比较大的影响, 如果准确率一直上不去,可以考虑调这个参数
                      'l2': 1e-9,  # l2正则化
                      'activate_classify': 'sigmoid',  # 'sigmoid',  # 最后一个layer, 即分类激活函数
                      'loss': 'binary_crossentropy',  # 损失函数, 可能有问题, 可以自己定义
                      # 'metrics': 'top_k_categorical_accuracy',  # 1070个类, 太多了先用topk,  这里数据k设置为最大:33
                      'metrics': 'accuracy',  # 保存更好模型的评价标准
                      'is_training': True,  # 训练后者是测试模型
                      'model_path': path_model,
                      # 模型地址, loss降低则保存的依据, save_best_only=True, save_weights_only=True
                      'path_hyper_parameters': path_hyper_parameters,  # 模型(包括embedding)，超参数地址,
                      'path_fineture': path_fineture,  # 保存embedding trainable地址, 例如字向量、词向量、bert向量等
                      },
            'embedding': {'layer_indexes': [12],  # bert取的层数
                          # 'corpus_path': '',     # embedding预训练数据地址,不配则会默认取conf里边默认的地址, keras-bert可以加载谷歌版bert,百度版ernie(需转换，https://github.com/ArthurRizar/tensorflow_ernie),哈工大版bert-wwm(tf框架，https://github.com/ymcui/Chinese-BERT-wwm)
                          },
            'data': {'train_data': "./data/train.csv",  # 训练数据
                     'val_data': "./data/val.csv",  # 验证数据
                     },
        }
    # 删除先前存在的模型和embedding微调模型等
    delete_file(path_model_dir)
    time_start = time.time()
    # graph初始化
    graph = Graph(hyper_parameters)
    print("graph init ok!")
    ra_ed = graph.word_embedding
    # 数据预处理
    pt = PreprocessTextMulti()
    print(ra_ed,rate)
    x_train, y_train,_,_ = pt.preprocess_label_ques_to_idx(hyper_parameters['embedding_type'],
                                                             hyper_parameters['data']['train_data'],
                                                             ra_ed, rate=rate, shuffle=True)
    print('train data progress ok!')
    x_val, y_val,_,_ = pt.preprocess_label_ques_to_idx(hyper_parameters['embedding_type'],
                                                         hyper_parameters['data']['val_data'],
                                                         ra_ed, rate=rate, shuffle=True)
    print("data progress ok!")
    print(len(y_train))
    # 训练
    graph.fit(x_train, y_train, x_val, y_val)
    print("耗时:" + str(time.time() - time_start))


def evaluate(path_hyper_parameter=path_hyper_parameters, rate=1.0):
    # 输入预测
    # 加载超参数
    hyper_parameters = load_json(path_hyper_parameter)
    pt = PreprocessTextMulti()
    # 模式初始化和加载
    graph = Graph(hyper_parameters)
    graph.load_model()
    ra_ed = graph.word_embedding
    # get validation data
    ques_list, val_list, que, val = pt.preprocess_label_ques_to_idx(hyper_parameters['embedding_type'],
                                                                    hyper_parameters['data']['val_data'],
                                                                    ra_ed, rate=rate, shuffle=True)
    print(len(ques_list))
    print("que:",len(que))
    # print(val)

    # str to token
    ques_embed_list = []
    count = 0
    acc_count = 0
    not_none_count = 0
    not_none_acc_count = 0
    sum_iou = 0
    sum_all_iou=0
    for index, que___ in enumerate(que):
        # print("原句 ", index, que[index])
        # print("真实分类 ", index, val[index])
        # print("ques: ", ques)
        ques_embed = ra_ed.sentence2idx(que[index])
        if hyper_parameters['embedding_type'] == 'albert':
            x_val_1 = np.array([ques_embed[0]])
            x_val_2 = np.array([ques_embed[1]])
            ques_embed = [x_val_1, x_val_2]
        else:
            x_val = ques_embed
        # print("ques_embed: ", ques_embed)
        if hyper_parameters['embedding_type'] == 'bert':
            x_val_1 = np.array([ques_embed[0]])
            x_val_2 = np.array([ques_embed[1]])
            x_val = [x_val_1, x_val_2]
        else:
            x_val = ques_embed
        # print("x_val", x_val)
        ques_embed_list.append(x_val)
        # 预测
        pred = graph.predict(x_val)
        # print(pred)
        # 取id to label and pred
        pre = pt.prereocess_idx(pred[0])
        # print("pre",pre)
        ls_nulti = []
        threshold = 0.44
        top_threshold = 0
        for i, ls in enumerate(pre[0]):
            if i == 0 or ls[1] > threshold:
                ls_nulti.append(ls)
                top_threshold = ls[1]
            elif abs(ls[1] - top_threshold) < top_threshold / 4.0:
                ls_nulti.append(ls)
        # print("预测结果", index, pre[0])
        # print(ls_nulti)
        res = cal_acc(ls_nulti, val[index].split(","))
        res_iou,res_all_iou = cal_iou(ls_nulti, val[index].split(","))
        sum_iou += res_iou
        sum_all_iou+=res_all_iou
        if res:
            if val[index] != "无":
                not_none_acc_count += 1
            acc_count += 1
        else:
            print("原句 ", index, que[index])
            print("真实分类 ", index, val[index])
            print("pre ", pre)
            print("iou ", res_iou)
        count += 1
        if val[index] != "无":
            not_none_count += 1
    print("acc: ", acc_count / count)
    print("not none acc: ", not_none_acc_count / not_none_count)
    print("average iou: ", sum_iou / sum_all_iou)
    # log
    append_log(hyper_parameters, acc_count / count, not_none_acc_count / not_none_count, threshold)


def pred_input(str_input, path_hyper_parameter=path_hyper_parameters):
    # 输入预测
    # 加载超参数
    hyper_parameters = load_json(path_hyper_parameter)
    pt = PreprocessTextMulti()
    # 模式初始化和加载
    graph = Graph(hyper_parameters)
    graph.load_model()
    ra_ed = graph.word_embedding
    ques = str_input
    # str to token
    ques_embed = ra_ed.sentence2idx(ques)
    if hyper_parameters['embedding_type'] == 'bert':
        x_val_1 = np.array([ques_embed[0]])
        x_val_2 = np.array([ques_embed[1]])
        x_val = [x_val_1, x_val_2]
    else:
        x_val = ques_embed
    # 预测
    pred = graph.predict(x_val)
    print(pred)
    # 取id to label and pred
    pre = pt.prereocess_idx(pred[0])
    ls_nulti = []
    for ls in pre[0]:
        if ls[1] >= 0.73:
            ls_nulti.append(ls)
    print(str_input)
    print(pre[0])
    print(ls_nulti)


def cal_acc(predict, ground):
    for item_pre in predict:
        flag = False
        for item_ground in ground:
            if item_pre[0] == item_ground:
                flag = True
                break
        if not flag:
            return False
    if len(ground) != len(predict):
        return False
    return True


def cal_iou(predict, ground):
    predict_ = [i[0] for i in predict]
    # get union
    union_list = set(predict_).union(set(ground))
    # get intersection
    intersect_list = set(predict_).intersection(set(ground))
    # return len(intersect_list) / len(union_list)
    return len(intersect_list),len(union_list)


if __name__ == "__main__":
    # train()

    # my test
    # print("##############################")
    # inp=["脑桥","双侧额叶","左侧脑干","左基底节区","双侧基底节","双侧侧脑室旁、右侧基底节区可见点片状低密度影，脑室、脑池及脑沟显示良好，中线结构居中"]
    # for i in inp:
    #     pred_input(i)
    # print("##############################")

    evaluate(path_hyper_parameters)
