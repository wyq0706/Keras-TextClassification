from openpyxl import load_workbook
from sklearn.model_selection import train_test_split

# get data
from sklearn.utils import shuffle

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
for index, row in enumerate(booksheet.rows):
    line = [col.value for col in row]
    if index == 0:
        f_labels = open('./data/labels.csv', 'w', encoding='utf-8')
        labels = ["无"]
        title = line[2].split(" ")[1:]
        for item in title:
            if item == "脑干" or item == "脑桥":
                labels.append("\n" + item)
                continue
            labels.append("\n左" + item)
            labels.append("\n右" + item)
        f_labels.writelines(labels)
        f_labels.close()
        continue
    if line[2] is not None and len(line[2].strip()) != 0:
        label_str = ",".join(i for i in line[2].strip().split(" ") if i != "")
        if label_str == "无":
            sum_list_is_none.append("\n" + "|,|".join([label_str, "".join(line[1].strip().split("\n"))]))
        else:
            # intensify
            sum_list_not_none.append("\n" + "|,|".join([label_str, "".join(line[1].strip().split("\n"))]))
            sum_list_not_none.append("\n" + "|,|".join([label_str, "".join(line[1].strip().split("\n"))]))

import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
train_is_none, val_is_none = train_test_split(sum_list_is_none, test_size=0.5, random_state=seed)
# improve ratio of labels that are not 无
train_not_none, val_not_none = train_test_split(sum_list_not_none, test_size=0.3, random_state=seed)

lines_train_has_title.append("|,|".join(["label", "ques"]))
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
