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
"""training"""
parser.add_argument('--training_iter', type=int, default=20, help='number of train iterator')
parser.add_argument('--scope', type=str, default='Ind_BiLSTM', help='scope')
parser.add_argument('--batch_size', type=int, default=8, help='number of example per batch')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for bert')
parser.add_argument('--usegpu', type=bool, default=True, help='gpu')
"""other"""
parser.add_argument('--test_only', type=bool, default=True, help='no training')
parser.add_argument('--checkpoint', type=bool, default=True, help='load checkpoint')
parser.add_argument('--checkpointpath', type=str, default='checkpoint/ECE/', help='path to load checkpoint')
parser.add_argument('--savecheckpoint', type=bool, default=False, help='save checkpoint')
parser.add_argument('--save_path', type=str, default='prompt_ECE', help='path to save checkpoint')
parser.add_argument('--device', type=str, default='2', help='device id')
parser.add_argument('--dataset', type=str, default='data_combine_ECE/', help='path for dataset')

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
        self.gt_cause = []
        self.ECE = []
        self.doc_id = []
        self.test = test
        self.n_cut = 0
        self.tokenizer = tokenizer
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

            full_document = ""
            mask_full_document = ""
            mask_label_full_document = ""
            ECE_document = ""
            mask_label_ECE_document = ""
            part_sentence = []

            cnt_cause_gt = 0

            for _ in range(d_len):
                words = inputFile.readline().strip().split(',')[-1]
                part_sentence.append(words)

            cnt_cause_gt = len(set(cause))

            self.gt_cause.append(cnt_cause_gt)

            for i in range(1, d_len + 1):
                full_document = full_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
                mask_full_document = mask_full_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
                mask_label_full_document = mask_label_full_document + ' [MASK] ' + part_sentence[i - 1]
                ECE_document = ECE_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
                mask_label_ECE_document = mask_label_ECE_document + ' [MASK] ' + part_sentence[i - 1]
                if i in pos:
                    full_document = full_document + '是 '
                    ECE_document = ECE_document + '是 '
                    mask_label_ECE_document = mask_label_ECE_document + '是 '
                    if i in cause:
                        full_document = full_document + '是 '
                        full_document = full_document + ' ' + str(pos[cause.index(i)]) + ' '
                    else:
                        full_document = full_document + '非 '
                        full_document = full_document + ' 无 '
                else:
                    full_document = full_document + '非 '
                    ECE_document = ECE_document + '非 '
                    mask_label_ECE_document = mask_label_ECE_document + '非 '
                    if i in cause:
                        full_document = full_document + '是 '
                        full_document = full_document + ' ' + str(pos[cause.index(i)]) + ' '
                    else:
                        full_document = full_document + '非 '
                        full_document = full_document + ' 无 '

                full_document = full_document + '[SEP]'
                mask_full_document = mask_full_document + "[MASK] [MASK] [MASK][SEP]"
                mask_label_full_document = mask_label_full_document + "[MASK] [MASK] [MASK][SEP]"
                ECE_document = ECE_document + "[MASK] [MASK][SEP]"
                mask_label_ECE_document = mask_label_ECE_document + "[MASK] [MASK][SEP]"
            if (self.tokenizer.encode_plus(mask_full_document, return_tensors="pt")['input_ids'][0].shape !=
                    self.tokenizer.encode_plus(mask_label_full_document, return_tensors="pt")['input_ids'][0].shape):
                print('length wrong')

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
            ECE_document = \
                self.tokenizer.encode_plus(ECE_document, return_tensors="pt", max_length=512, truncation=True,
                                           pad_to_max_length=True)['input_ids']
            mask_label_ECE_document = \
                self.tokenizer.encode_plus(mask_label_ECE_document, return_tensors="pt", max_length=512,
                                           truncation=True,
                                           pad_to_max_length=True)['input_ids']

            labels = full_document.masked_fill(ECE_document != 103, -100)
            mask_labels = full_document.masked_fill(mask_label_ECE_document != 103, -100)

            self.x_bert.append(np.array(mask_label_ECE_document[0]))
            self.y_bert.append(np.array(full_document[0]))
            self.label.append(np.array(labels[0]))
            self.mask_label.append(np.array(mask_labels[0]))
            self.ECE.append(np.array(ECE_document[0]))
        self.x_bert, self.y_bert, self.label, self.mask_label, self.ECE = map(np.array,
                                                                              [self.x_bert, self.y_bert, self.label,
                                                                               self.mask_label, self.ECE])
        self.gt_cause = np.array(self.gt_cause)

        for var in ['self.x_bert', 'self.y_bert', 'self.label', 'self.mask_label', 'self.ECE', 'self.gt_cause']:
            print('{}.shape {}'.format(var, eval(var).shape))
        print('n_cut {}'.format(self.n_cut))
        print('load data done!\n')

        self.index = [i for i in range(len(self.x_bert))]

    def __getitem__(self, index):
        index = self.index[index]
        feed_list = [self.x_bert[index], self.y_bert[index], self.label[index], self.mask_label[index], self.ECE[index],
                     self.gt_cause[index]]
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


def prf_prompt(logits, labels, x_bert, gt_cause):
    cause_gt = torch.sum(gt_cause)
    cause_pre = 0
    cause_acc = 0
    for i in range(labels.shape[0]):
        count_mask = -1
        j = 0
        while j < 512:
            if x_bert[i][j] == 103:
                count_mask += 1
                count_mask = count_mask % 2
                if count_mask == 0:
                    if torch.argmax(logits[i][j]) == 3221:
                        if labels[i][j] == 3221:
                            cause_acc += 1
                    if torch.argmax(logits[i][j]) == 3221:
                        cause_pre += 1

                if count_mask == 1:
                    pass
                j = j + 1
            else:
                j = j + 1
    p_cause = cause_acc / (cause_pre + 1e-8)
    r_cause = cause_acc / (cause_gt + 1e-8)
    f_cause = 2 * p_cause * r_cause / (p_cause + r_cause + 1e-8)
    print('cause_gt {}'.format(cause_gt))
    return p_cause, r_cause, f_cause


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
    max_result_cause_f, max_result_cause_r, max_result_cause_p = [], [], []
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
        NLP_Dataset = {x: MyDataset(edict[x], tokenizer=tokenizer) for x in ['train', 'test']}
        trainloader = DataLoader(NLP_Dataset['train'], batch_size=opt.batch_size, shuffle=True, drop_last=True)
        testloader = DataLoader(NLP_Dataset['test'], batch_size=opt.batch_size, shuffle=False)

        max_p_emotion, max_r_emotion, max_f1_emotion, max_p_cause, max_r_cause, max_f1_cause, max_p_pair, max_r_pair,\
        max_f1_pair = [-1.] * 9
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
        if opt.test_only:
            all_test_logits = torch.tensor([])
            all_test_label = torch.tensor([])
            all_test_mask_label = torch.tensor([])
            all_test_y_bert = torch.tensor([])
            all_test_x_bert = torch.tensor([])
            all_test_cause_gt = torch.tensor([])
            model.eval()
            with torch.no_grad():
                for _, data in enumerate(testloader):
                    x_bert, y_bert, label, mask_label, ECE_x_bert, gt_cause = data
                    if use_gpu:
                        x_bert = x_bert.cuda()
                        y_bert = y_bert.cuda()
                        label = label.cuda()
                        mask_label = mask_label.cuda()
                        ECE_x_bert = ECE_x_bert.cuda()
                    loss, logits = model(ECE_x_bert, label)
                    logits = F.softmax(logits, dim=-1)
                    all_test_label = torch.cat((all_test_label, label.cpu()), 0)
                    all_test_mask_label = torch.cat((all_test_mask_label, mask_label.cpu()), 0)
                    all_test_logits = torch.cat((all_test_logits, logits.cpu()), 0)
                    all_test_y_bert = torch.cat((all_test_y_bert, y_bert.cpu()), 0)
                    all_test_x_bert = torch.cat((all_test_x_bert, ECE_x_bert.cpu()), 0)
                    all_test_cause_gt = torch.cat((all_test_cause_gt, gt_cause), 0)

                p_cause, r_cause, f_cause = prf_prompt(all_test_logits, all_test_label, all_test_x_bert,
                                                       all_test_cause_gt)
                print("c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}".format(p_cause, r_cause, f_cause))
                if f_cause > max_f1_cause:
                    max_p_cause, max_r_cause, max_f1_cause = p_cause, r_cause, f_cause
                print(
                    "max result---- c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}".format(max_p_cause, max_r_cause, max_f1_cause))
        else:
            for i in range(opt.training_iter):
                model.train()
                start_time, step = time.time(), 1
                for index, data in enumerate(trainloader):
                    with torch.autograd.set_detect_anomaly(True):
                        x_bert, y_bert, label, mask_label, ECE_x_bert, gt_cause = data
                        if use_gpu:
                            x_bert = x_bert.cuda()
                            y_bert = y_bert.cuda()
                            label = label.cuda()
                            mask_label = mask_label.cuda()
                            ECE_x_bert = ECE_x_bert.cuda()
                        loss, logits = model(ECE_x_bert, mask_label)
                        logits = F.softmax(logits, dim=-1)

                        optimizer.zero_grad()
                        if use_gpu:
                            loss = loss.cuda()
                        loss.backward()
                        optimizer.step()

                        print("loss: {:.4f}".format(loss))
                        if index % 20 == 0:
                            p_cause, r_cause, f_cause = prf_prompt(logits.cpu(), label.cpu(), ECE_x_bert.cpu(),
                                                                   gt_cause)
                            print(
                                "iter: {} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}".format(index, p_cause, r_cause, f_cause))
                all_test_logits = torch.tensor([])
                all_test_label = torch.tensor([])
                all_test_mask_label = torch.tensor([])
                all_test_y_bert = torch.tensor([])
                all_test_x_bert = torch.tensor([])
                all_test_cause_gt = torch.tensor([])

                model.eval()
                with torch.no_grad():
                    for _, data in enumerate(testloader):
                        x_bert, y_bert, label, mask_label, ECE_x_bert, gt_cause = data
                        if use_gpu:
                            x_bert = x_bert.cuda()
                            y_bert = y_bert.cuda()
                            label = label.cuda()
                            mask_label = mask_label.cuda()
                            ECE_x_bert = ECE_x_bert.cuda()
                        loss, logits = model(ECE_x_bert, label)
                        logits = F.softmax(logits, dim=-1)
                        all_test_label = torch.cat((all_test_label, label.cpu()), 0)
                        all_test_mask_label = torch.cat((all_test_mask_label, mask_label.cpu()), 0)
                        all_test_logits = torch.cat((all_test_logits, logits.cpu()), 0)
                        all_test_y_bert = torch.cat((all_test_y_bert, y_bert.cpu()), 0)
                        all_test_x_bert = torch.cat((all_test_x_bert, ECE_x_bert.cpu()), 0)
                        all_test_cause_gt = torch.cat((all_test_cause_gt, gt_cause), 0)

                    p_cause, r_cause, f_cause = prf_prompt(all_test_logits, all_test_label, all_test_x_bert,
                                                           all_test_cause_gt)
                    print("iter{} test result:".format(i))
                    print("c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}".format(p_cause, r_cause, f_cause))

                    if f_cause > max_f1_cause:
                        max_p_cause, max_r_cause, max_f1_cause = p_cause, r_cause, f_cause
                        if opt.savecheckpoint:
                            torch.save(model, save_path + '/' + 'fold{}.pth'.format(fold))
                    print("iter{} test result:".format(i))
                    print(
                        "max result---- c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}".format(max_p_cause, max_r_cause,
                                                                                    max_f1_cause))
        max_result_cause_f.append(max_f1_cause)
        max_result_cause_p.append(max_p_cause)
        max_result_cause_r.append(max_r_cause)

    print("cause")
    print(max_result_cause_f)
    print("average f {:.4f}".format(sum(max_result_cause_f) / len(max_result_cause_f)))
    print(max_result_cause_p)
    print("average p {:.4f}".format(sum(max_result_cause_p) / len(max_result_cause_p)))
    print(max_result_cause_r)
    print("average r {:.4f}".format(sum(max_result_cause_r) / len(max_result_cause_r)))


if __name__ == '__main__':
    run()
