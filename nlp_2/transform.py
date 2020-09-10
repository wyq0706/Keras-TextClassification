# coding=utf-8
from sklearn.utils import shuffle
from openpyxl import load_workbook
from sklearn.model_selection import train_test_split
import numpy

def reverse_side(s):
    rtn_s = ""
    for index, item in enumerate(s):
        if item == "左":
            rtn_s += "右"
        elif item == "右":
            rtn_s += "左"
        else:
            rtn_s += item
    return rtn_s


def isNone(labels):
    return len(labels)==1 and labels[0]=='无'

def getLabel(ground):
    vec=[]
    if ground[0] == '0':
        vec.append("多发")
    elif ground[0] == '1':
        vec.append("散发")
    else:
        vec.append("无")
    if ground[1]=='1':
        if ground[2]=='1':
            vec.append("低密度点状")
        if ground[3]=='1':
            vec.append("低密度条状")
        if ground[4]=='1':
            vec.append("低密度片状")
        if ground[5]=='1':
            vec.append("低密度其他形状")
    if ground[6]=='1':
        if ground[7] == '1':
            vec.append("高密度点状")
        if ground[8] == '1':
            vec.append("高密度条状")
        if ground[9] == '1':
            vec.append("高密度片状")
        if ground[10] == '1':
            vec.append("高密度其他形状")
    if ground[11]=='1':
        if ground[12] == '1':
            vec.append("混杂密度点状")
        if ground[13] == '1':
            vec.append("混杂密度条状")
        if ground[14] == '1':
            vec.append("混杂密度片状")
        if ground[15] == '1':
            vec.append("混杂密度其他形状")
    if ground[16]=='1':
        if ground[17] == '1':
            vec.append("无密度点状")
        if ground[18] == '1':
            vec.append("无密度条状")
        if ground[19] == '1':
            vec.append("无密度片状")
        if ground[20] == '1':
            vec.append("无密度其他形状")
    return vec


def readData():
    in_file=['./data/all_txt_0.txt','./data/all_txt_1.txt','./data/all_txt_2.txt','./data/all_txt_3.txt']
    res_data=[]
    all_count=0
    diff_count=0
    has_dot_low=0
    count=0
    res_data_util=[]
    for filename in in_file:
        with open(filename,"r",encoding="gb2312") as f:
            for index, item in enumerate(f.readlines()):
                # print(item.strip())
                all_count+=1
                split_res=item.strip().split("|")
                labels=split_res[0].split(',')
                sentence=split_res[1]
                last_ground_truth=split_res[2]
                if len(sentence) >160:
                    print("oversize",len(sentence),sentence)
                if not isNone(labels):
                    for index_,item_ in enumerate(labels):
                        count+=1
                        ground_truth=split_res[2+index_]
                        if ground_truth!=last_ground_truth:
                            print("hh",labels,sentence)
                            diff_count+=1
                            # pass
                        res_labels=','.join(getLabel(ground_truth))
                        if len(getLabel(ground_truth))==1:
                            print("??",labels,sentence)
                        else:
                            add_str='|,|'.join([res_labels, '('+item_+')'+sentence]) + '\n'
                            if ground_truth[2]=='1':
                                has_dot_low+=1
                            if add_str.find("左") != -1 or add_str.find("右") != -1 and ground_truth[2]=='0':
                                # sum_list_is_none.append(reverse_side(add_str))
                                res_data_util.append(reverse_side(add_str))
                            res_data.append(add_str)
    print(diff_count/count)
    print("low dot: ",has_dot_low/count)
    return res_data,res_data_util

def writeData(data,util):
    f_train = open('./data/train.csv', 'w', encoding='utf-8')
    f_val = open('./data/val.csv', 'w', encoding='utf-8')
    f_test = open('./data/test.csv', 'w', encoding='utf-8')
    seed = 7
    numpy.random.seed(seed)
    train_txt, tmp_txt = train_test_split(data, test_size=0.4, random_state=seed)
    val_txt,test_txt=train_test_split(tmp_txt,test_size=0.5,random_state=int(seed/2))
    f_train.writelines(["label|,|ques\n"])
    f_val.writelines(["label|,|ques\n"])
    f_test.writelines(["label|,|ques\n"])
    f_train.writelines(train_txt)
    f_train.writelines(util)
    f_val.writelines(val_txt)
    f_test.writelines(test_txt)
    f_train.close()
    f_val.close()
    f_test.close()

def run():
    res,res_util=readData()
    writeData(res,res_util)


if __name__ == "__main__":
    run()
