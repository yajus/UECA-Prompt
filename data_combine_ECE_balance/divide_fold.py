# encoding: utf-8
import os
import random

new_complete_file = open("all_data_pair_ECE_balance.txt", 'r', encoding='utf-8')
lines = new_complete_file.readlines()
doc_content = {}
doc_id = 0
i = 0
while i < len(lines):
    if lines[i] == "":
        break
    doclen = int(lines[i].split(" ")[1])
    content_list = []
    content_list.append(lines[i])
    content_list.append(lines[i + 1])
    for j in range(doclen):
        content_list.append(lines[i + 2 + j])
    doc_content[doc_id] = content_list
    doc_id += 1
    i += doclen + 2

# random.shuffle(doc_content)

fold_size = int(len(doc_content) / 10)
for fold in range(1, 11):
    fold_train = open("fold{}_train.txt".format(fold), 'w', encoding='utf-8')
    fold_test = open("fold{}_test.txt".format(fold), 'w', encoding='utf-8')
    mark_out = {}
    for j in range((fold - 1) * fold_size, (fold) * fold_size):
        content_list = doc_content[j]
        # fold_test.write("{} {} {} {}\n".format(j+1, len(content_list)-2,
        #                                        content_list[0].split()[2], content_list[0].split()[3]))
        for l in range(0, len(content_list)):
            fold_test.write(content_list[l])
        mark_out[j] = 1
    for j in range(len(doc_content)):
        if j not in mark_out:
            content_list = doc_content[j]
            # fold_train.write("{} {} {} {}\n".format(j+1, len(content_list)-2,
            #                                         content_list[0].split()[2], content_list[0].split()[3]))
            for l in range(0, len(content_list)):
                fold_train.write(content_list[l])
