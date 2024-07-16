import torch
from torch import nn, autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel

class EarlyDecHan(nn.Module):
    def __init__(self, hparams):
        super(EarlyDecHan, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hparams.get('hidden_size', 256)
        self.embedding_size = hparams.get('embedding_size', 768)
        self.output_size = hparams.get('output_size', 1)
        self.dropout = hparams.get('dropout', 0.1)
        self.batch_size = hparams.get('batch_size', 1)

        # Bert
        self.tokenizer = BertTokenizer.from_pretrained("/home/yuzhezi/Models/bert-base-uncased")
        self.bert = BertModel.from_pretrained("/home/yuzhezi/Models/bert-base-uncased")

        # LSTM
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, 1, batch_first=True, bidirectional=True)

        # Attention
        self.query = nn.Parameter(torch.Tensor(1, self.hidden_size * 2), requires_grad=True)
        nn.init.uniform_(self.query, -1.0 / self.hidden_size, 1.0 / self.hidden_size)
        self.attention = nn.MultiheadAttention(self.hidden_size * 2, 1, batch_first=True, dropout=self.dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def init_hidden(self):
        return (
            autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_size)).to(self.device),
            autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_size)).to(self.device)
        )

    def forward(self, input_ids, attention_mask, lengths):
        self.hidden = self.init_hidden()
        X = self.bert(input_ids=input_ids, attention_mask=attention_mask)['pooler_output'].view(self.batch_size, max(lengths), self.embedding_size)
        X = pack_padded_sequence(X, torch.tensor(lengths), batch_first=True, enforce_sorted=False)
        X, _ = self.lstm(X, self.hidden)
        X, _ = pad_packed_sequence(X, batch_first=True)
        key_padding_mask = torch.zeros(self.batch_size, max(lengths)).to(self.device)
        for i, mask in enumerate(lengths):
            key_padding_mask[i, :mask] = 1
        X, _ = self.attention(self.query.expand(self.batch_size, 1, -1), X, X, key_padding_mask=key_padding_mask)
        X = torch.squeeze(X)
        return self.classifier(X)
