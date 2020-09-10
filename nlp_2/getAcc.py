

def detect(sentence,illness):
    res={
        "scope":0,
        "low_density":[0,0,0,0],
        "high_density":[0,0,0,0],
        "both_density":[0,0,0,0],
        "none_density":[0,0,0,0]
    }
    if sentence.find("散发")!=-1:
        res["scope"]=1
    elif sentence.find("多发")!=-1:
        res["scope"]=0
    else:
        res["scope"]=2

def isNone(labels):
    return len(labels)==1 and labels[0]=='无'


def run():
    with open('./data/all_txt_0.txt',"r",encoding="gb2312") as f:
        count=0
        acc_count=0
        for index, item in enumerate(f.readlines()):
            print(item.strip())
            split_res=item.strip().split("|")
            labels=split_res.split(',')[0]
            if not isNone(labels):
                count+=1
                flag=True
                for index_,item_ in labels:
                    ground_truth=split_res[3+index_]
                    predict_truth=detect(split_res[1],item_)
                    if predict_truth!=ground_truth:
                        flag=False
                    print(item_,predict_truth,ground_truth)
                if flag:
                    acc_count+=1
            if index>20:
                break
        print(acc_count/count)


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


if __name__ == "__main__":
    run()