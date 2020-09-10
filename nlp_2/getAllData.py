from sklearn.utils import shuffle
from openpyxl import load_workbook
from sklearn.model_selection import train_test_split

# get data

workbook = load_workbook(u'/home/liuaohan/sentence label.xlsx')
booksheet = workbook.active

f_train = open('./data/train.csv', 'w', encoding='utf-8')
f_val = open('./data/val.csv', 'w', encoding='utf-8')
lines_train = []
lines_val = []
lines_train_has_title = []
lines_val_has_title = []

sum_list_is_none = []
sum_list_not_none = []

add_none_count = 0
add_not_none_count = 0
origin_none_count = 0
origin_not_none_count = 0


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


def replace_both_side(s):
    rtn_l = s
    rtn_r = s
    return rtn_l.replace("双侧", "左侧"), rtn_r.replace("双侧", "右侧")


lines_train_has_title.append("|,|".join(["label", "ques"]))

for index, row in enumerate(booksheet.rows):
    line = [col.value for col in row]
    if index == 0:
        f_labels = open('./data/labels.csv', 'w', encoding='utf-8')
        labels = ["无"]
        title = line[2].split(" ")[1:]
        for item in title:
            if item == "脑干" or item == "脑桥":
                labels.append("\n" + item)
                # lines_train_has_title.append("\n"+"|,|".join([item,item]))
                continue
            labels.append("\n左" + item)
            # lines_train_has_title.append("\n"+"|,|".join(["左"+item, "左侧"+item]))
            # lines_train_has_title.append("\n" + "|,|".join(["左" + item, "左" + item]))
            labels.append("\n右" + item)
            # lines_train_has_title.append("\n"+"|,|".join(["右" + item, "右侧" + item]))
            # lines_train_has_title.append("\n" + "|,|".join(["右" + item, "右" + item]))
            # lines_train_has_title.append("\n" + "|,|".join(["右" + item+","+"左"+item, "双侧" + item]))
            # lines_train_has_title.append("\n" + "|,|".join(["右" + item + "," + "左" + item, "两侧" + item]))
        # lines_train_has_title.append("\n左基底节区,右基底节区|,|双侧基底节")
        # lines_train_has_title.append("\n右额叶,右顶叶,右枕叶|,|右侧额顶枕叶")
        # lines_train_has_title.append("\n左额叶,左顶叶,左枕叶|,|左侧额顶枕叶")
        # lines_train_has_title.append("\n右枕叶,右顶叶|,|左侧顶枕叶")
        # lines_train_has_title.append("\n左枕叶,左顶叶|,|右侧顶枕叶")
        f_labels.writelines(labels)
        f_labels.close()
        continue
    if line[2] is not None and len(line[2].strip()) != 0:
        label_str = ",".join(i for i in line[2].strip().split(" ") if i != "")
        if label_str == "无":
            add_str = "\n" + "|,|".join([label_str, "".join(line[1].strip().split("\n"))])
            sum_list_is_none.append(add_str)
            origin_none_count += 1
            # data augmentation_0
            if origin_none_count % 3 == 1:
                if add_str.find("左") != -1 or add_str.find("右") != -1:
                    # sum_list_is_none.append(reverse_side(add_str))
                    lines_train_has_title.append(reverse_side(add_str))
                    add_none_count += 1
            if origin_none_count % 3 == 2:
                if add_str.find("双侧") != -1:
                    rtn_l, rtn_r = replace_both_side(add_str)
                    # sum_list_is_none.append(rtn_l)
                    # sum_list_is_none.append(rtn_r)
                    lines_train_has_title.append(rtn_l)
                    lines_train_has_title.append(rtn_r)
                    add_none_count += 2
        else:
            # if len(label_str) == 1 and label_str[0][0] == "左" or label_str[0][0] == "右":
            #     # data augmentation_0
            #     # the model easily predicts both sides of organisms
            #     sum_list_not_none.append("\n" + "|,|".join([label_str, "".join(line[1].strip().split("\n"))]))
            # data augmentation_1
            add_str = "\n" + "|,|".join([label_str, "".join(line[1].strip().split("\n"))])
            sum_list_not_none.append(add_str)
            origin_not_none_count += 1
            # if origin_not_none_count % 3 != 0:
            sum_list_not_none.append(reverse_side(add_str))
            lines_train_has_title.append(reverse_side(add_str))
            add_not_none_count += 1
            # sum_list_not_none.append("\n" + "|,|".join([label_str, "".join(line[1].strip().split("\n"))]))

# data augmentation_2
# sum_list_not_none.append("\n左枕叶,左基底节区,右基底节区|,|左枕叶及双侧基底节区见多发低密度影，边界清")
# sum_list_not_none.append("\n左额叶,左颞叶|,|左侧额叶、颞叶见不规则高、低混杂密度影")
# sum_list_not_none.append("\n左额叶,左颞叶|,|左侧额、颞叶见片状低密度影，左侧额、颞部颅板下方见线形稍高密度影，邻近见引流管影")
# sum_list_not_none.append("\n左半卵圆中心|,|左侧半卵园中心区及侧脑室旁白质内见小斑片低密度影，边界尚清")
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# keep ratio of not none data the same
train_is_none, val_is_none = train_test_split(sum_list_is_none, test_size=0.28, random_state=seed)
# improve ratio of labels that are not 无
train_not_none, val_not_none = train_test_split(sum_list_not_none, test_size=0.28, random_state=seed)

lines_val_has_title.append("|,|".join(["label", "ques"]))

lines_train.extend(train_is_none)
lines_train.extend(train_not_none)
lines_train_shuffled = shuffle(lines_train)
lines_train_has_title.extend(lines_train_shuffled)

lines_val.extend(val_is_none)
lines_val.extend(val_not_none)
lines_val_shuffled = shuffle(lines_val)
lines_val_has_title.extend(lines_val_shuffled)

f_train.writelines(lines_train_has_title)
f_val.writelines(lines_val_has_title)

f_train.close()
f_val.close()

print("add_none_count: ", add_none_count, add_none_count / origin_none_count)
print("add_not_none_count: ", add_not_none_count, add_not_none_count / origin_not_none_count)