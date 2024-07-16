import random

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from transformers import BertModel


def seq_loss(log_sequence_probs, pred_logits, target, mask):
    log_sequence_probs, sequence_probs = log_sequence_probs.view(-1), torch.exp(log_sequence_probs)
    gold_pred_probs = torch.softmax(pred_logits, dim=-1)[:, :, target.item()].squeeze(dim=-1)
    gold_pred_probs = torch.cat(
        [gold_pred_probs, torch.tensor([-1e-3], device=gold_pred_probs.device).unsqueeze(dim=0)],
        dim=-1
    ).view(-1)
    sequence_probs = sequence_probs.view(-1)
    gold_pred_probs = gold_pred_probs.view(-1)
    mask = mask.view(-1)
    l = -torch.log(torch.sum(sequence_probs * gold_pred_probs * mask))
    return l


def delay_loss(action_probs, pred_probs, o=50):
    action_probs = action_probs.squeeze(dim=0)
    delay_loss_value = 0
    L = action_probs.shape[0]
    for i, posability in enumerate(action_probs):
        delay_loss_value += (posability[0]) * (i)
    return delay_loss_value / L



def full_loss(final_logits, target):
    final_logits = final_logits.reshape((-1, 2))
    return F.cross_entropy(final_logits, target.reshape((-1)))


def compute_log_p_tragic(action_probs, drop_out=0.):
    seq_len = action_probs.size(1)

    b_curr = torch.zeros_like(action_probs[:, 0, :])
    b_next = torch.zeros_like(action_probs[:, 0, :])

    p = random.random()
    b_curr[:, 0] = 0 if p < drop_out else torch.log(action_probs[:, 0, 0])
    b_curr[:, 1] = 0 if p < drop_out else torch.log(action_probs[:, 0, 1])
    b_values = [b_curr[:, 1].view(1)]
    mask = [0 if p < drop_out else 1]
    for i in range(1, seq_len):
        p = random.random()
        b_next[:, 0] = b_curr[:, 0] if p < drop_out else torch.log(action_probs[:, i, 0]) + b_curr[:, 0]
        b_next[:, 1] = b_curr[:, 0] if p < drop_out else torch.log(action_probs[:, i, 1]) + b_curr[:, 0]
        mask.append(0 if p < drop_out else 1)
        b_values.append(b_next[:, 1].view(1))
        b_curr, b_next = b_next.clone(), b_curr.clone()
        if i == seq_len - 1:
            b_values.append(b_next[:, 0].view(1))
            mask.append(1)
    mask_tensor = torch.tensor(mask).to(b_values[0].dtype).to(b_values[0].device)
    b = torch.stack(b_values, dim=0).reshape((-1, 1))
    return b, mask_tensor


def all_loss_fn(action_logits, pred_logits, target):
    tragic_log_probs, mask = compute_log_p_tragic(torch.softmax(action_logits, dim=-1))
    loss = seq_loss(tragic_log_probs, pred_logits, target, mask)
    loss += full_loss(pred_logits[:, -1, :], target)
    if target.item() == 1:
        action_probs = torch.softmax(action_logits, dim=-1)
        loss = loss + delay_loss(action_probs) * 5e-2

    return loss


class StepAttention(nn.Module):
    def __init__(self, express_size):
        super(StepAttention, self).__init__()
        self.to_K = nn.Linear(in_features=express_size, out_features=express_size)
        self.query = nn.Linear(in_features=express_size, out_features=1, bias=False)
        torch.nn.init.xavier_uniform_(self.to_K.weight, gain=1)
        torch.nn.init.xavier_uniform_(self.query.weight, gain=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, value):
        texts_K = torch.tanh(self.to_K(value))  # (batch_size,seq_len,768)
        scores = self.query(texts_K)  # (batch_size,seq_len,1)
        ans = []
        for i in range(1, value.shape[1] + 1):
            texts_feature = value[:, 0:i, :]
            temp_scores = torch.softmax(scores[:, 0:i, :], dim=1)
            ans.append(torch.bmm(temp_scores.permute((0, 2, 1)), texts_feature))  # batch_size,1,768
        ans = torch.cat(ans, dim=1)
        return ans


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    mean_pooled = sum_embeddings / input_mask_expanded.sum(1)
    return mean_pooled


def truncate_after_first_one(actions):
    first_two_idx = (actions == 1).nonzero(as_tuple=True)[0]
    if first_two_idx.numel() == 0:
        return actions
    truncated_actions = actions[:first_two_idx[0] + 1]
    return truncated_actions


def max_action(action_probs):
    _, action_list = torch.max(action_probs.squeeze(dim=0), dim=-1)
    action_list = truncate_after_first_one(action_list)
    if 1 not in action_list:
        action_list[-1] = 1
    return action_list


class EarlyDecEsdm(nn.Module):
    def __init__(self, hparams):
        super(EarlyDecEsdm, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hparams.get('hidden_size', 256)
        self.embedding_size = hparams.get('embedding_size', 768)
        self.output_size = hparams.get('output_size', 1)
        self.dropout = hparams.get('dropout', 0.1)
        self.batch_size = hparams.get('batch_size', 1)
        
        # Define BERT model
        self.bert = BertModel.from_pretrained('/home/yuzhezi/Master/Test/ESDM/model_hub/bert-base-uncased')
        self.dropout = nn.Dropout(0.2)

        # Define prediction layers
        self.get_pred = nn.LSTM(input_size=768, hidden_size=768, batch_first=True)
        self.pred_attention = StepAttention(768)
        self.pc1 = nn.Linear(768, self.hidden_size)
        self.pc2 = nn.Linear(self.hidden_size, self.output_size * 2)

        # Define action layers
        self.get_action = nn.LSTM(input_size=770, hidden_size=770, batch_first=True)
        self.act_attention = StepAttention(770)
        self.ac1 = nn.Linear(770, self.hidden_size)
        self.ac2 = nn.Linear(self.hidden_size, self.output_size * 2)

        # Initialize weights
        for layer in [self.get_pred, self.get_action, self.pc1, self.pc2, self.ac1, self.ac2]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)

    def make_action_probs(self, hidden_state):
        hidden_state = self.get_action(hidden_state)[0] + hidden_state
        action_logits = self.ac2(torch.relu(self.ac1(self.dropout(hidden_state))))
        return hidden_state, action_logits

    def make_prediction(self, hidden_state):
        hidden_state = self.get_pred(hidden_state)[0] + hidden_state
        hidden_state = self.pred_attention(hidden_state)
        pred_logits = self.pc2(torch.relu(self.pc1(self.dropout(hidden_state))))
        return hidden_state, pred_logits

    def forward(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.squeeze(dim=0), attention_mask.squeeze(dim=0)
        hidden_state = mean_pooling(self.bert(input_ids=input_ids, attention_mask=attention_mask), attention_mask)
        hidden_state = hidden_state.unsqueeze(dim=0)
        hidden_state, pred_logits = self.make_prediction(hidden_state)
        a_hidden_state = torch.cat([hidden_state, pred_logits], dim=-1)
        _, action_logits = self.make_action_probs(a_hidden_state)
        return action_logits, pred_logits
