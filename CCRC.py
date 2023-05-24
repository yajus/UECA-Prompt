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
parser.add_argument('--checkpointpath', type=str, default='checkpoint/CCRC/', help='path to load checkpoint')
parser.add_argument('--savecheckpoint', type=bool, default=False, help='save checkpoint')
parser.add_argument('--save_path', type=str, default='prompt_CCRC', help='path to save checkpoint')
parser.add_argument('--device', type=str, default='2', help='device id')
parser.add_argument('--dataset', type=str, default='data_combine_CCRC/', help='path for dataset')
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
        self.gt_conditional = []
        self.emotion_index = []
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
            result_label = int(line[2])
            pairs = eval('[' + inputFile.readline().strip() + ']')
            pos, cause = zip(*pairs)

            if len(set(pos)) != 1:
                print("wrong")

            self.emotion_index.append(pos[0])

            full_document = ""
            mask_Conditional_document = ""
            mask_label_Conditional_document = ""
            part_sentence = []

            for i in range(d_len):
                words = inputFile.readline().strip().split(',')[-1]
                part_sentence.append(words)

            self.gt_conditional.append(result_label)

            for i in range(1, d_len + 1):
                full_document = full_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
                mask_Conditional_document = mask_Conditional_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
                mask_label_Conditional_document = mask_label_Conditional_document + ' [MASK] ' + part_sentence[i - 1]
                if i in pos:
                    full_document = full_document + '是 '
                    mask_Conditional_document = mask_Conditional_document + '是 '
                    mask_label_Conditional_document = mask_label_Conditional_document + '是 '
                    if i in cause:
                        full_document = full_document + '是 '
                        mask_Conditional_document = mask_Conditional_document + '是 '
                        mask_label_Conditional_document = mask_label_Conditional_document + '是 '
                        if result_label == 1:
                            full_document = full_document + ' ' + str(pos[cause.index(i)]) + ' '
                        else:
                            full_document = full_document + ' 无 '
                    else:
                        full_document = full_document + '非 '
                        mask_Conditional_document = mask_Conditional_document + '非 '
                        mask_label_Conditional_document = mask_label_Conditional_document + '非 '
                        full_document = full_document + ' 无 '
                else:
                    full_document = full_document + '非 '
                    mask_Conditional_document = mask_Conditional_document + '非 '
                    mask_label_Conditional_document = mask_label_Conditional_document + '非 '
                    if i in cause:
                        full_document = full_document + '是 '
                        mask_Conditional_document = mask_Conditional_document + '是 '
                        mask_label_Conditional_document = mask_label_Conditional_document + '是 '
                        if result_label == 1:
                            full_document = full_document + ' ' + str(pos[cause.index(i)]) + ' '
                        else:
                            full_document = full_document + ' 无 '
                    else:
                        full_document = full_document + '非 '
                        full_document = full_document + ' 无 '
                        mask_Conditional_document = mask_Conditional_document + '非 '
                        mask_label_Conditional_document = mask_label_Conditional_document + '非 '

                full_document = full_document + '[SEP]'
                mask_Conditional_document = mask_Conditional_document + "[MASK][SEP]"  ### 用于输入
                mask_label_Conditional_document = mask_label_Conditional_document + "[MASK][SEP]"  ### 用于获取训练输入label

            if (self.tokenizer.encode_plus(full_document, return_tensors="pt")['input_ids'][0].shape !=
                    self.tokenizer.encode_plus(mask_Conditional_document, return_tensors="pt")['input_ids'][0].shape):
                print('length wrong')

            full_document = \
                self.tokenizer.encode_plus(full_document, return_tensors="pt", max_length=512, truncation=True,
                                           pad_to_max_length=True)['input_ids']
            mask_Conditional_document = \
                self.tokenizer.encode_plus(mask_Conditional_document, return_tensors="pt", max_length=512,
                                           truncation=True, pad_to_max_length=True)['input_ids']
            mask_label_Conditional_document = \
                self.tokenizer.encode_plus(mask_label_Conditional_document, return_tensors="pt", max_length=512,
                                           truncation=True, pad_to_max_length=True)['input_ids']

            labels = full_document.masked_fill(mask_Conditional_document != 103, -100)
            mask_labels = full_document.masked_fill(mask_label_Conditional_document != 103, -100)
            self.y_bert.append(np.array(full_document[0]))
            self.label.append(np.array(labels[0]))
            self.mask_label.append(np.array(mask_labels[0]))
            self.x_bert.append(np.array(mask_Conditional_document[0]))
        self.x_bert, self.y_bert, self.label, self.mask_label, self.emotion_index = map(
            np.array,
            [self.x_bert,
             self.y_bert,
             self.label,
             self.mask_label,
             self.emotion_index])
        self.gt_conditional = np.array(self.gt_conditional)

        for var in ['self.x_bert', 'self.y_bert', 'self.label', 'self.mask_label', 'self.gt_conditional']:
            print('{}.shape {}'.format(var, eval(var).shape))
        print('n_cut {}'.format(self.n_cut))
        print('load data done!\n')

        self.index = [i for i in range(len(self.y_bert))]

    def __getitem__(self, index):
        index = self.index[index]
        feed_list = [self.x_bert[index], self.y_bert[index], self.label[index], self.mask_label[index],
                     self.gt_conditional[index], self.emotion_index[index]]
        return feed_list

    def __len__(self):
        return len(self.y_bert)


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


def prf_prompt(logits, labels, mask_full_document, gt_conditional, emotion_index):
    pre_conditional = []
    label_index = [122, 123, 124, 125, 126, 127, 128, 129, 130, 8108, 8111, 8110, 8124, 8122, 8115, 8121, 8126, 8123,
                   8131, 8113, 8128, 8130, 8133, 8125, 8132, 8153, 8149, 8143, 8162, 8114, 8176, 8211, 8226, 8229, 8198,
                   8216, 8234, 8218, 8240, 8164, 8245, 8239, 8250, 8252, 8208, 8248, 8264, 8214, 8249, 8145, 8246, 8247,
                   8251, 8267, 8222, 8259, 8272, 8255, 8257, 8183, 8398, 8356, 8381, 8308, 8284, 8347, 8369, 8360, 8419,
                   8203, 8459, 8325, 8454, 8473, 8273]
    Conditional_gt = torch.sum(gt_conditional)
    Conditional_pre = 0
    Conditional_acc = 0
    for i in range(labels.shape[0]):
        count_pridict = 0
        j = 0
        count_sentence = 0
        count_positive = 0
        while j < 512:
            if labels[i][j] != -100:
                count_sentence += 1
                mark_cause = mask_full_document[i][j - 1]
                mask = torch.zeros([21128])
                case = [label_index[k] for k in range(max(0, -opt.window_size + count_sentence - 1),
                                                      min(75, opt.window_size + count_sentence))]
                case.append(3187)
                for index in case:
                    mask[index] = 1
                if mark_cause == 3221:
                    count_pridict += 1
                    logits_ = torch.argmax(logits[i][j] * mask)
                    if logits_ in label_index and logits_ == label_index[emotion_index[i] - 1]:
                        count_positive += 1

                j = j + 1
            else:
                j = j + 1
        if count_positive / (count_pridict + 1e-8) > 0.5:
            Conditional_pre += 1
            pre_conditional.append(1)
            if gt_conditional[i] == 1:
                Conditional_acc += 1
        else:
            pre_conditional.append(0)

    p_cause = Conditional_acc / (Conditional_pre + 1e-8)
    r_cause = Conditional_acc / (Conditional_gt + 1e-8)
    f_cause = 2 * p_cause * r_cause / (p_cause + r_cause + 1e-8)
    print(
        'Conditional_gt {} Conditional_pre {} Conditional_acc {} lenofdata {} '.format(Conditional_gt, Conditional_pre,
                                                                                       Conditional_acc, len(labels)))
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

    max_result_conditional_f = []
    max_result_conditional_p = []
    max_result_conditional_r = []

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

        max_p_conditional, max_r_conditional, max_f1_conditional = [-1.] * 3
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
        if opt.test_only:
            all_test_logits = torch.tensor([])
            all_test_label = torch.tensor([])
            all_test_mask_label = torch.tensor([])
            all_test_y_bert = torch.tensor([])
            all_test_conditional_gt = torch.tensor([])
            all_test_emotion_index = torch.tensor([])
            all_test_x_bert = torch.tensor([])
            model.eval()
            with torch.no_grad():
                for _, data in enumerate(testloader):
                    x_bert, y_bert, label, mask_label, gt_conditional, emotion_index = data
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
                    all_test_conditional_gt = torch.cat((all_test_conditional_gt, gt_conditional), 0)
                    all_test_emotion_index = torch.cat((all_test_emotion_index, emotion_index), 0)
                p_Conditional, r_Conditional, f_Conditional = prf_prompt(all_test_logits, all_test_label,
                                                                         all_test_x_bert,
                                                                         all_test_conditional_gt,
                                                                         all_test_emotion_index.int())
                print("c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}".format(p_Conditional, r_Conditional, f_Conditional))

                if f_Conditional > max_f1_conditional:
                    max_p_conditional, max_r_conditional, max_f1_conditional =\
                        p_Conditional, r_Conditional, f_Conditional
                    torch.save(model, save_path + '/' + 'fold{}.pth'.format(fold))
                print(
                    "max result---- c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}".format(max_p_conditional, max_r_conditional,
                                                                                max_f1_conditional))

        else:
            for i in range(opt.training_iter):
                model.train()
                start_time, step = time.time(), 1
                for index, data in enumerate(trainloader):
                    with torch.autograd.set_detect_anomaly(True):
                        x_bert, y_bert, label, mask_label, gt_conditional, emotion_index = data
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
                            p_Conditional, r_Conditional, f_Conditional = prf_prompt(logits.cpu(), label.cpu(),
                                                                                     x_bert.cpu(), gt_conditional,
                                                                                     emotion_index)
                            print("iter: {} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}".format(index, p_Conditional,
                                                                                        r_Conditional,
                                                                                        f_Conditional))
                all_test_logits = torch.tensor([])
                all_test_label = torch.tensor([])
                all_test_mask_label = torch.tensor([])
                all_test_y_bert = torch.tensor([])
                all_test_conditional_gt = torch.tensor([])
                all_test_emotion_index = torch.tensor([])
                all_test_x_bert = torch.tensor([])

                model.eval()
                with torch.no_grad():
                    for _, data in enumerate(testloader):
                        x_bert, y_bert, label, mask_label, gt_conditional, emotion_index = data
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
                        all_test_conditional_gt = torch.cat((all_test_conditional_gt, gt_conditional), 0)
                        all_test_emotion_index = torch.cat((all_test_emotion_index, emotion_index), 0)

                    p_Conditional, r_Conditional, f_Conditional = prf_prompt(all_test_logits, all_test_label,
                                                                             all_test_x_bert,
                                                                             all_test_conditional_gt,
                                                                             all_test_emotion_index.int())
                    print("iter{} test result:".format(i))
                    print("c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}".format(p_Conditional, r_Conditional, f_Conditional))

                    if f_Conditional > max_f1_conditional:
                        max_p_conditional, max_r_conditional, max_f1_conditional =\
                            p_Conditional, r_Conditional, f_Conditional
                        if opt.savecheckpoint:
                            torch.save(model, save_path + '/' + 'fold{}.pth'.format(fold))

                    print("iter{} test result:".format(i))
                    print(
                        "max result---- c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}".format(max_p_conditional,
                                                                                    max_r_conditional,
                                                                                    max_f1_conditional))
        max_result_conditional_f.append(max_f1_conditional)
        max_result_conditional_p.append(max_p_conditional)
        max_result_conditional_r.append(max_r_conditional)

    print("conditional")
    print(max_result_conditional_f)
    print("average f {:.4f}".format(sum(max_result_conditional_f) / len(max_result_conditional_f)))
    print(max_result_conditional_p)
    print("average p {:.4f}".format(sum(max_result_conditional_p) / len(max_result_conditional_p)))
    print(max_result_conditional_r)
    print("average r {:.4f}".format(sum(max_result_conditional_r) / len(max_result_conditional_r)))


if __name__ == '__main__':
    run()
