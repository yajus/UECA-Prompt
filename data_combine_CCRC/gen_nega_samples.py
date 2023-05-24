# encoding: utf-8
import numpy as np


def write_a_doc(ofile, doc_num, label, cond_label, doclen, line1, emo, cau, emo_list, cau_list, con_list):
    ofile.write("{} {} {} {}\n".format(doc_num, doclen, label, cond_label))
    ofile.write(line1)
    caucnt = 0
    concnt = 0
    for i in range(doclen):
        if i + 1 in emo:
            ori_line = emo_list[0].strip().split(",")
            ofile.write("{},{},{},{}\n".format(i + 1, ori_line[1], ori_line[2], ori_line[3]))
        elif i + 1 in cau:
            ofile.write(cau_list[caucnt])
            caucnt += 1
        else:
            if concnt < len(con_list):
                ori_line = con_list[concnt].strip().split(",")[3]
                concnt += 1
                ofile.write("{},null,null,{}\n".format(i + 1, ori_line))
            else:
                ofile.write("{},null,null,\n".format(i + 1))


data = open("data.txt", 'r', encoding='utf-8').readlines()
ofile = open("data_wneg.txt", 'w', encoding='utf-8')
# if you want to create the dataset with different n, change the following values
n = 2

doc_content = {}
doc_id = 0
i = 0
while i < len(data):
    if data[i] == "":
        break
    doclen = int(data[i].split(" ")[1])
    content_list = []
    content_list.append(data[i])
    content_list.append(data[i + 1])
    pairs = eval('[' + data[i + 1].strip() + ']')
    emo, cau = zip(*pairs)
    content_list.append(emo)
    content_list.append(cau)
    emo_list = []
    cau_list = []
    con_list = []
    for j in range(doclen):
        if j + 1 in emo:
            emo_list.append(data[i + 2 + j])
            if j + 1 in cau:
                cau_list.append(data[i + 2 + j])
        elif j + 1 in cau:
            cau_list.append(data[i + 2 + j])
        else:
            con_list.append(data[i + 2 + j])

    content_list.append(emo_list)
    content_list.append(cau_list)
    content_list.append(con_list)

    doc_content[doc_id] = content_list
    doc_id += 1
    i += doclen + 2

doc_num = 1
for doc_id in range(len(doc_content)):
    content_list = doc_content[doc_id]
    [line0, line1, emo, cau, emo_list, cau_list, con_list] = content_list
    emoword = emo_list[0].split(",")[1]
    doclen = int(line0.strip().split(" ")[1])
    label = int(line0.strip().split(" ")[2])
    mark_nega_index = {}
    # generate original samples
    write_a_doc(ofile, doc_num, 1, label, doclen, line1, emo, cau, emo_list, cau_list, con_list)
    doc_num += 1

    # generate negative samples
    mark_nega_index = {}
    for cnt in range(n):
        nega_index = np.random.randint(len(doc_content))
        while (nega_index in mark_nega_index):
            nega_index = np.random.randint(len(doc_content))
        mark_nega_index[nega_index] = 1

        nega_con_list = doc_content[nega_index][6]
        write_a_doc(ofile, doc_num, 1 - label, label, doclen, line1, emo, cau, emo_list, cau_list, nega_con_list)
        doc_num += 1

    mark_nega_index = {}
    for cnt in range(n):
        nega_index = np.random.randint(len(doc_content))
        negaemoword = doc_content[nega_index][4][0].split(",")[1]
        while (nega_index in mark_nega_index or negaemoword == emoword):
            nega_index = np.random.randint(len(doc_content))
            negaemoword = doc_content[nega_index][4][0].split(",")[1]
        mark_nega_index[nega_index] = 1

        nega_emo_list = doc_content[nega_index][4]
        write_a_doc(ofile, doc_num, 0, label, doclen, line1, emo, cau, nega_emo_list, cau_list, con_list)
        doc_num += 1
