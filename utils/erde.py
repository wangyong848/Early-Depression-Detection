import numpy as np
from torch import nn
from tqdm import tqdm

import math
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

cfp = 0.1296
cfn = 1
ctp = 1
ctn = 0
# 真阳第一次发现，假阳概率
first_time_tp = list()
max_tp = list()
max_fp = list()


def lc0k(target, result, trajectory: list, o=50):
    time, result = 0, result.item()
    for i in range(0, len(trajectory)):
        if trajectory[i] == 1:
            time = 1 - 1 / (1 + math.exp(i - o))
            break
    if result == 0 and target == 0:
        return ctn
    elif result == 1 and target == 1:
        return ctp * time
    elif result == 1 and target == 0:
        return cfp
    elif result == 0 and target == 1:
        return cfn


'''    
    if label == 1 and true_label == 1:
        return delay_cost(k=delay, break_point=_o) * _c_tp
    elif label == 1 and true_label == 0:
        return _c_fp
    elif label == 0 and true_label == 1:
        return _c_fn
    elif label == 0 and true_label == 0:
        return 0

def erde(labels_list, true_labels_list, delay_list, c_fp=0.1296, c_tp=1, c_fn=1, o=5):
'''


def calculate_ERDE(model, dataset, kind='train'):
    model.eval()
    datasetloader = DataLoader(dataset, batch_size=1, shuffle=False)
    ans = [list() for _ in range(0, len(dataset))]
    y = list()
    ans_p = [list() for _ in range(0, len(dataset))]
    with torch.no_grad():
        torch.autograd.set_detect_anomaly(True)
        for slice in range(1, 10):
            id = 0
            # input_ids.long(), attention_mask, first_noun, third_noun, time.long(), ss3_word,topic, label,name
            for input_ids, attention_mask, first_noun, third_noun, time_hour, ss3_word, topic, label, _ in tqdm(
                    datasetloader):
                # if x.shape[1] - x.shape[1] % 7<=6:continue
                # x, y = x.resize_((1, x.shape[1] - x.shape[1] % 7, 128)).cuda(), y.cuda()
                input_ids, attention_mask, first_noun, third_noun, time_hour, label = input_ids.cuda(), attention_mask.cuda(), first_noun.cuda(), third_noun.cuda(), time_hour.cuda(), label.cuda()
                input_ids, attention_mask, time_hour, first_noun, third_noun = input_ids.squeeze(
                    dim=0)[0:slice], attention_mask.squeeze(dim=0)[0:slice], time_hour.squeeze(dim=0)[0:slice], first_noun.squeeze(
                    dim=0)[0:slice], third_noun.squeeze(dim=0)[0:slice]
                topic = topic.squeeze(dim=0)[0:slice].cuda()
                if slice != 1 and slice != 2 and slice != 3 and slice != 150:
                    res = F.softmax(
                        model(input_ids, attention_mask, first_noun, time_hour),
                        dim=0)
                else:
                    res = F.softmax(
                        model(input_ids, attention_mask, first_noun, time_hour, True),
                        dim=0)
                p = res
                res = torch.argmax(res.reshape((-1)), dim=0, keepdim=False).detach().cpu().item()
                ans[id].append(res)
                ans_p[id].append(p.reshape((-1))[res].cpu().item())
                if len(y) < len(dataset):
                    y.append(label.detach().cpu().item())
                else:
                    if y[id] != label.detach().cpu().item():
                        print(y[id])
                        print(label)
                id = id + 1
        ERDE_50, id, ERDE_5 = 0, 0, 0
        predict = list()
        for id in range(0, len(y)):
            result = sum(ans[id])
            if result == 0:
                predict.append(0)
            else:
                predict.append(1)
            if result == 0 and y[id] == 0:
                continue
            elif result != 0 and y[id] == 1:
                ERDE_50 = ERDE_50 + ctp * lc0k(ans[id], o=50, p=ans_p[id])
                ERDE_5 = ERDE_5 + ctp * lc0k(ans[id], o=5, p=ans_p[id])
            elif result != 0 and y[id] == 0:
                ERDE_50 = ERDE_50 + cfp
                ERDE_5 = ERDE_5 + cfp
                max_fp.append(max(ans_p[id]))
            elif result == 0 and y[id] == 1:
                ERDE_50 = ERDE_50 + cfn
                ERDE_5 = ERDE_5 + cfn
        y = np.asarray(y)
        predict = np.asarray(predict)
        print(f1_score(predict, y))
        print(accuracy_score(predict, y))
        print(recall_score(predict, y))
        print(precision_score(predict, y))
        print(ERDE_5, ERDE_50)
        print(max_fp)
        print(max_tp)
        print(first_time_tp)
        print(ERDE_5 / len(dataset), ERDE_50 / len(dataset))
