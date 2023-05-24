# encoding: utf-8

data = open("original_ecpe.txt", 'r', encoding='utf-8').readlines()
labels = open("final_labels.txt", 'r').readlines()
ofile = open("data.txt", 'w', encoding='utf-8')

i = 0
while i < len(data):
    docid = int(data[i].split()[0])
    doclen = int(data[i].split()[1])
    ofile.write("%s %s\n" % (data[i].strip(), labels[docid - 1].strip()))
    ofile.write(data[i + 1])
    for j in range(doclen):
        ofile.write(data[i + 2 + j])
    i += doclen + 2

ofile.close()
