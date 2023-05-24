import argparse
import os
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
import time
import numpy as np

"""setting agrparse"""
parser = argparse.ArgumentParser(description='Training')

"""model struct"""
parser.add_argument('--n_hidden', type=int, default=100, help='number of hidden unit')
parser.add_argument('--n_class', type=int, default=2, help='number of distinct class')
parser.add_argument('--window_size', type=int, default=2, help='size of the emotion cause pair window')
parser.add_argument('--feature_layer', type=int, default=3, help='number of layer iterations')
parser.add_argument('--log_file_name', type=str, default='log', help='name of log file')
parser.add_argument('--model_type', type=str, default='ISML', help='type of model')
parser.add_argument('--num_for_M', type=int, default=2, help='for M2M module')
"""training"""
parser.add_argument('--training_iter', type=int, default=20, help='number of train iterator')
parser.add_argument('--scope', type=str, default='Ind_BiLSTM', help='scope')
parser.add_argument('--batch_size', type=int, default=8, help='number of example per batch')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for bert')
parser.add_argument('--usegpu', type=bool, default=True, help='gpu')
"""other"""
parser.add_argument('--test_only', type=bool, default=False, help='no training')
parser.add_argument('--checkpoint', type=bool, default=False, help='load checkpoint')
parser.add_argument('--checkpointpath', type=str, default='checkpoint/ECPE/', help='path to load checkpoint')
parser.add_argument('--savecheckpoint', type=bool, default=True, help='save checkpoint')
parser.add_argument('--save_path', type=str, default='prompt_ECPE', help='path to save checkpoint')
parser.add_argument('--device', type=str, default='2', help='device id')
parser.add_argument('--dataset', type=str, default='data_combine_ECPE/', help='path for dataset')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

if opt.usegpu and torch.cuda.is_available():
    use_gpu = True


def print_time():
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))


class MyDataset(Dataset):
    def __init__(self, input_file, test=False, tokenizer=None):
        print('load data_file: {}'.format(input_file))
        self.x_bert, self.y_bert, self.label, self.mask_label = [], [], [], []
        self.gt_emotion, self.gt_cause, self.gt_pair = [], [], []
        self.doc_id = []
        self.test = test
        self.n_cut = 0
        self.tokenizer = tokenizer
        cnt_over_limit = 0
        inputFile = open(input_file, 'r')
        while True:
            line = inputFile.readline()
            if line == '':
                break
            line = line.strip().split()
            self.doc_id.append(line[0])
            d_len = int(line[1])
            pairs = eval('[' + inputFile.readline().strip() + ']')
            pos, cause = zip(*pairs)

            diction = {}
            # count the max num of emotion for one cause
            for i in set(pairs):
                if i[1] in diction.keys():
                    diction[i[1]].append(i[0])
                else:
                    diction[i[1]] = [i[0]]

            full_document = ""
            mask_full_document = ""
            mask_label_full_document = ""
            part_sentence = []
            cnt_emotion_gt = 0
            cnt_cause_gt = 0
            cnt_pair_gt = 0

            for _ in range(d_len):
                words = inputFile.readline().strip().split(',')[-1]
                part_sentence.append(words)
            cnt_emotion_gt = len(set(pos))
            cnt_cause_gt = len(set(cause))
            cnt_pair_gt = len(set(pairs))
            self.gt_emotion.append(cnt_emotion_gt)
            self.gt_pair.append(cnt_pair_gt)
            self.gt_cause.append(cnt_cause_gt)
            for i in range(1, d_len + 1):
                full_document = full_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
                mask_full_document = mask_full_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
                mask_label_full_document = mask_label_full_document + ' [MASK] ' + part_sentence[i - 1]
                if i in pos:
                    full_document = full_document + '是 '
                    if i in cause:
                        full_document = full_document + '是 '
                        for j in range(2):
                            if j < len(diction[i]):
                                full_document = full_document + ' ' + str(diction[i][j]) + ' '
                            else:
                                full_document = full_document + ' 无 '
                    else:
                        full_document = full_document + '非 '
                        full_document = full_document + ' 无 无 '
                else:
                    full_document = full_document + '非 '
                    if i in cause:
                        full_document = full_document + '是 '
                        for j in range(opt.num_for_M):
                            if j < len(diction[i]):
                                full_document = full_document + ' ' + str(diction[i][j]) + ' '
                            else:
                                full_document = full_document + ' 无 '
                    else:
                        full_document = full_document + '非 '
                        full_document = full_document + ' 无 无'

                full_document = full_document + '[SEP]'
                mask_full_document = mask_full_document + "[MASK] [MASK] [MASK][MASK][SEP]"
                mask_label_full_document = mask_label_full_document + "[MASK] [MASK] [MASK][MASK][SEP]"
            if (self.tokenizer.encode_plus(mask_full_document, return_tensors="pt")['input_ids'][0].shape !=
                    self.tokenizer.encode_plus(full_document, return_tensors="pt")['input_ids'][0].shape):
                print('length wrong')

            count_len = len(self.tokenizer.encode_plus(mask_full_document, return_tensors="pt")['input_ids'][0])
            if count_len > 512:
                print("Over limit length{} document{}".format(count_len, line[0]))
                cnt_over_limit += 1
            mask_full_document = \
                self.tokenizer.encode_plus(mask_full_document, return_tensors="pt", max_length=512, truncation=True,
                                           pad_to_max_length=True)['input_ids']
            full_document = \
                self.tokenizer.encode_plus(full_document, return_tensors="pt", max_length=512, truncation=True,
                                           pad_to_max_length=True)['input_ids']
            mask_label_full_document = \
                self.tokenizer.encode_plus(mask_label_full_document, return_tensors="pt", max_length=512,
                                           truncation=True,
                                           pad_to_max_length=True)['input_ids']
            labels = full_document.masked_fill(mask_full_document != 103, -100)
            mask_labels = full_document.masked_fill(mask_label_full_document != 103, -100)
            self.x_bert.append(np.array(mask_full_document[0]))  # A[MASK]情感句，[MASK]原因句[MASK][SEP]
            self.y_bert.append(np.array(full_document[0]))  # A(是/非)情感句，(有/无)原因句(#*/[PAD])
            self.label.append(np.array(labels[0]))  # -100(是/非)-100(有/无)-100(#*/[PAD])
            self.mask_label.append(np.array(mask_labels[0]))
        self.x_bert, self.y_bert, self.label, self.mask_label = map(np.array, [self.x_bert, self.y_bert, self.label,
                                                                               self.mask_label])
        self.gt_emotion, self.gt_cause, self.gt_pair = map(np.array, [self.gt_emotion, self.gt_cause, self.gt_pair])
        for var in ['self.x_bert', 'self.y_bert', 'self.label', 'self.mask_label', 'self.gt_emotion', 'self.gt_cause',
                    'self.gt_pair']:
            print('{}.shape {}'.format(var, eval(var).shape))
        print('n_cut {}'.format(self.n_cut))
        print('load data done!\n')

        self.index = [i for i in range(len(self.y_cause))]
        print("num_for_over_limit{}".format(cnt_over_limit))

    def __getitem__(self, index):
        index = self.index[index]
        feed_list = [self.x_bert[index], self.y_bert[index], self.label[index], self.mask_label[index],
                     self.gt_emotion[index], self.gt_cause[index], self.gt_pair[index]]
        return feed_list

    def __len__(self):
        return len(self.x_bert)


class prompt_bert(torch.nn.Module):
    def __init__(self, bert_path='./bert-base-chinese'):
        super(prompt_bert, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert.resize_token_embeddings(len(self.tokenizer))

    def forward(self, x_bert, labels):
        output = self.bert(x_bert, labels=labels)
        loss, logits = output.loss, output.logits
        return loss, logits


def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{}'.format(
        opt.batch_size, opt.learning_rate))
    print('training_iter-{}\n'.format(opt.training_iter))


def prf_prompt(logits, labels, x_bert, gt_emotion, gt_cause, gt_pair):
    label_index = [122, 123, 124, 125, 126, 127, 128, 129, 130, 8108, 8111, 8110, 8124, 8122, 8115, 8121, 8126, 8123,
                   8131, 8113, 8128, 8130, 8133, 8125, 8132, 8153, 8149, 8143, 8162, 8114, 8176, 8211, 8226, 8229, 8198,
                   8216, 8234, 8218, 8240, 8164, 8245, 8239, 8250, 8252, 8208, 8248, 8264, 8214, 8249, 8145, 8246, 8247,
                   8251, 8267, 8222, 8259, 8272, 8255, 8257, 8183, 8398, 8356, 8381, 8308, 8284, 8347, 8369, 8360, 8419,
                   8203, 8459, 8325, 8454, 8473, 8273]
    emo_gt = torch.sum(gt_emotion)
    emo_pre = 0
    emo_acc = 0
    cause_gt = torch.sum(gt_cause)
    cause_pre = 0
    cause_acc = 0
    pair_gt = torch.sum(gt_pair)
    pair_pre = 0
    pair_acc = 0
    for i in range(labels.shape[0]):
        count_mask = -1
        j = 0
        count_sentence = 0
        while j < 512:
            if x_bert[i][j] == 103:
                count_mask += 1
                count_mask = count_mask % 3
                if count_mask == 0:
                    if labels[i][j] == 3221:
                        if torch.argmax(logits[i][j]) == 3221:
                            emo_acc += 1
                    if torch.argmax(logits[i][j]) == 3221:
                        emo_pre += 1

                if count_mask == 1:
                    if torch.argmax(logits[i][j]) == 3221:
                        cause_pre += 1
                    if labels[i][j] == 3221:
                        if torch.argmax(logits[i][j]) == 3221:
                            cause_acc += 1

                if count_mask == 2:
                    count_sentence += 1
                    mask = torch.zeros([21128])
                    case = [label_index[k] for k in range(max(0, -opt.window_size + count_sentence - 1),
                                                          min(75, opt.window_size + count_sentence))]
                    case.append(3187)
                    logits_list = []
                    logits_gt_list = []
                    for index in case:
                        mask[index] = 1
                    logits_1 = torch.argmax(logits[i][j] * mask).tolist()
                    logits_list.append(logits_1)
                    logits_gt_list.append(labels[i][j].tolist())

                    j += 1
                    if j >= 512:
                        break
                    logits_2 = torch.argmax(logits[i][j] * mask).tolist()
                    logits_list.append(logits_2)
                    logits_gt_list.append(labels[i][j].tolist())
                    for k in set(logits_list):
                        if k in label_index:
                            pair_pre += 1
                            if k in set(logits_gt_list):
                                pair_acc += 1
                j = j + 1
            else:
                j = j + 1
    p_emotion = emo_acc / (emo_pre + 1e-8)
    p_cause = cause_acc / (cause_pre + 1e-8)
    p_pair = pair_acc / (pair_pre + 1e-8)
    r_emotion = emo_acc / (emo_gt + 1e-8)
    r_cause = cause_acc / (cause_gt + 1e-8)
    r_pair = pair_acc / (pair_gt + 1e-8)
    f_emotion = 2 * p_emotion * r_emotion / (p_emotion + r_emotion + 1e-8)
    f_cause = 2 * p_cause * r_cause / (p_cause + r_cause + 1e-8)
    f_pair = 2 * p_pair * r_pair / (p_pair + r_pair + 1e-8)
    print('emo_gt {}  cause_gt {}  pair_gt {}'.format(emo_gt, cause_gt, pair_gt))
    return p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair


def run():
    if opt.log_file_name:
        save_path = opt.save_path

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # sys.stdout = open(save_path + '/' + opt.log_file_name, 'w')

    print_time()
    bert_path = './bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    # train
    print_training_info()  # 输出训练的超参数信息

    max_result_emo_f, max_result_emo_p, max_result_emo_r = [], [], []
    max_result_pair_f, max_result_pair_p, max_result_pair_r = [], [], []
    max_result_cause_f, max_result_cause_p, max_result_cause_r = [], [], []
    for fold in range(1, 11):
        # model
        print('build model..')

        model = prompt_bert(bert_path)
        print('build model end...')
        if opt.checkpoint:
            model = torch.load(opt.checkpointpath + '/fold{}.pth'.format(fold),
                               map_location=torch.device('cpu'))
        if use_gpu:
            model = model.cuda()

        train_file_name = 'fold{}_train.txt'.format(fold)
        test_file_name = 'fold{}_test.txt'.format(fold)
        print('############# fold {} begin ###############'.format(fold))
        train = opt.dataset + train_file_name
        test = opt.dataset + test_file_name
        edict = {"train": train, "test": test}
        NLP_Dataset = {x: MyDataset(edict[x], test=(x == 'test'), tokenizer=tokenizer) for x in ['train', 'test']}
        trainloader = DataLoader(NLP_Dataset['train'], batch_size=opt.batch_size, shuffle=True, drop_last=True)
        testloader = DataLoader(NLP_Dataset['test'], batch_size=opt.batch_size, shuffle=False)

        max_p_emotion, max_r_emotion, max_f1_emotion, max_p_cause, max_r_cause, \
        max_f1_cause, max_p_pair, max_r_pair, max_f1_pair = [-1.] * 9
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

        if opt.test_only:
            all_test_logits = torch.tensor([])
            all_test_label = torch.tensor([])
            all_test_mask_label = torch.tensor([])
            all_test_y_bert = torch.tensor([])
            all_test_x_bert = torch.tensor([])
            all_test_emotion_gt = torch.tensor([])
            all_test_cause_gt = torch.tensor([])
            all_test_pair_gt = torch.tensor([])

            model.eval()
            with torch.no_grad():
                for _, data in enumerate(testloader):
                    x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair = data
                    if use_gpu:
                        x_bert = x_bert.cuda()
                        y_bert = y_bert.cuda()
                        label = label.cuda()
                        mask_label = mask_label.cuda()
                    loss, logits = model(x_bert, label)
                    logits = F.softmax(logits, dim=-1)
                    all_test_label = torch.cat((all_test_label, label.cpu()), 0)
                    all_test_mask_label = torch.cat((all_test_mask_label, mask_label.cpu()), 0)
                    all_test_logits = torch.cat((all_test_logits, logits.cpu()), 0)
                    all_test_y_bert = torch.cat((all_test_y_bert, y_bert.cpu()), 0)
                    all_test_x_bert = torch.cat((all_test_x_bert, x_bert.cpu()), 0)
                    all_test_emotion_gt = torch.cat((all_test_emotion_gt, gt_emotion), 0)
                    all_test_cause_gt = torch.cat((all_test_cause_gt, gt_cause), 0)
                    all_test_pair_gt = torch.cat((all_test_pair_gt, gt_pair), 0)

                p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair = prf_prompt(
                    all_test_logits, all_test_label, all_test_x_bert, all_test_emotion_gt, all_test_cause_gt,
                    all_test_pair_gt)
                print(
                    "e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f}"
                    " c_f: {:.4f} pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                        p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair))
                if f_emotion > max_f1_emotion:
                    max_f1_emotion, max_p_emotion, max_r_emotion = f_emotion, p_emotion, r_emotion
                if f_cause > max_f1_cause:
                    max_f1_cause, max_p_cause, max_r_cause = f_cause, p_cause, r_cause
                if f_pair > max_f1_pair:
                    max_f1_pair, max_p_pair, max_r_pair = f_pair, p_pair, r_pair
                print(
                    "max result---- e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f}"
                    " c_f: {:.4f} pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                        max_p_emotion, max_r_emotion, max_f1_emotion, max_p_cause, max_r_cause, max_f1_cause,
                        max_p_pair, max_r_pair, max_f1_pair))
        else:
            for i in range(opt.training_iter):
                model.train()
                start_time, step = time.time(), 1
                for index, data in enumerate(trainloader):
                    with torch.autograd.set_detect_anomaly(True):
                        x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair = data
                        if use_gpu:
                            x_bert = x_bert.cuda()
                            y_bert = y_bert.cuda()
                            label = label.cuda()
                            mask_label = mask_label.cuda()
                        loss, logits = model(x_bert, mask_label)
                        logits = F.softmax(logits, dim=-1)

                        optimizer.zero_grad()
                        if use_gpu:
                            loss = loss.cuda()
                        loss.backward()
                        optimizer.step()

                        print("loss: {:.4f}".format(loss))
                        if index % 20 == 0:
                            p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair,\
                            r_pair, f_pair = prf_prompt(logits.cpu(), label.cpu(), x_bert.cpu(),
                                                        gt_emotion, gt_cause, gt_pair)
                            print(
                                "iter: {} e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}"
                                " pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                                    index, p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair,
                                    f_pair))
                all_test_logits = torch.tensor([])
                all_test_label = torch.tensor([])
                all_test_mask_label = torch.tensor([])
                all_test_y_bert = torch.tensor([])
                all_test_x_bert = torch.tensor([])
                all_test_emotion_gt = torch.tensor([])
                all_test_cause_gt = torch.tensor([])
                all_test_pair_gt = torch.tensor([])

                model.eval()
                with torch.no_grad():
                    for _, data in enumerate(testloader):
                        x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair = data
                        if use_gpu:
                            x_bert = x_bert.cuda()
                            y_bert = y_bert.cuda()
                            label = label.cuda()
                            mask_label = mask_label.cuda()
                        loss, logits = model(x_bert, label)
                        logits = F.softmax(logits, dim=-1)
                        all_test_label = torch.cat((all_test_label, label.cpu()), 0)
                        all_test_mask_label = torch.cat((all_test_mask_label, mask_label.cpu()), 0)
                        all_test_logits = torch.cat((all_test_logits, logits.cpu()), 0)
                        all_test_y_bert = torch.cat((all_test_y_bert, y_bert.cpu()), 0)
                        all_test_x_bert = torch.cat((all_test_x_bert, x_bert.cpu()), 0)
                        all_test_emotion_gt = torch.cat((all_test_emotion_gt, gt_emotion), 0)
                        all_test_cause_gt = torch.cat((all_test_cause_gt, gt_cause), 0)
                        all_test_pair_gt = torch.cat((all_test_pair_gt, gt_pair), 0)

                    p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair = prf_prompt(
                        all_test_logits, all_test_label, all_test_x_bert, all_test_emotion_gt, all_test_cause_gt,
                        all_test_pair_gt)
                    print("iter{} test result:".format(i))
                    print(
                        "e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r:"
                        " {:.4f} c_f: {:.4f} pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                            p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair))
                    if f_emotion > max_f1_emotion:
                        max_f1_emotion, max_p_emotion, max_r_emotion = f_emotion, p_emotion, r_emotion
                    if f_cause > max_f1_cause:
                        max_f1_cause, max_p_cause, max_r_cause = f_cause, p_cause, r_cause
                    if f_pair > max_f1_pair:
                        max_f1_pair, max_p_pair, max_r_pair = f_pair, p_pair, r_pair
                        if opt.savecheckpoint:
                            torch.save(model, save_path + '/' + 'fold{}.pth'.format(fold))
                    print("iter{} test result:".format(i))
                    print(
                        "max result---- e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}"
                        " pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                            max_p_emotion, max_r_emotion, max_f1_emotion, max_p_cause, max_r_cause, max_f1_cause,
                            max_p_pair, max_r_pair, max_f1_pair))
        max_result_emo_f.append(max_f1_emotion)
        max_result_cause_f.append(max_f1_cause)
        max_result_pair_f.append(max_f1_pair)
        max_result_emo_p.append(max_p_emotion)
        max_result_cause_p.append(max_p_cause)
        max_result_pair_p.append(max_p_pair)
        max_result_emo_r.append(max_r_emotion)
        max_result_cause_r.append(max_r_cause)
        max_result_pair_r.append(max_r_pair)

    print("emotion")
    print(max_result_emo_f)
    print("average f {:.4f}".format(sum(max_result_emo_f) / len(max_result_emo_f)))
    print(max_result_emo_p)
    print("average p {:.4f}".format(sum(max_result_emo_p) / len(max_result_emo_p)))
    print(max_result_emo_r)
    print("average r {:.4f}".format(sum(max_result_emo_r) / len(max_result_emo_r)))
    print("cause")
    print(max_result_cause_f)
    print("average f {:.4f}".format(sum(max_result_cause_f) / len(max_result_cause_f)))
    print(max_result_cause_p)
    print("average p {:.4f}".format(sum(max_result_cause_p) / len(max_result_cause_p)))
    print(max_result_cause_r)
    print("average r {:.4f}".format(sum(max_result_cause_r) / len(max_result_cause_r)))
    print("pair")
    print(max_result_pair_f)
    print("average f {:.4f}".format(sum(max_result_pair_f) / len(max_result_pair_f)))
    print(max_result_pair_p)
    print("average p {:.4f}".format(sum(max_result_pair_p) / len(max_result_pair_p)))
    print(max_result_pair_r)
    print("average r {:.4f}".format(sum(max_result_pair_r) / len(max_result_pair_r)))


if __name__ == '__main__':
    run()
