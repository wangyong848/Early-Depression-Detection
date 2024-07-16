import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Collate:
    def __call__(self, batch):
        input_ids, attention_mask, labels = zip(*batch)
        lengths = [len(item[0]) for item in batch]
        return (
            pad_sequence(input_ids, batch_first=True),
            pad_sequence(attention_mask, batch_first=True),
            lengths,
            labels
        )


class EriskData(Dataset):
    def __init__(self, hparams, post_num=400, flag='train'):
        self.hparams = hparams
        assert flag in ['train', 'val', 'test'], 'ERROR!'
        self.flag = flag
        if self.hparams.get('max_len', 64) == 64:
            with open("/home/yuzhezi/Data/Erisk2017/2017_64.pkl", "rb") as f:
                data = pickle.load(f)
        else:
            with open("/home/yuzhezi/Data/Erisk2017/2017.pkl", "rb") as f:
                data = pickle.load(f)
        mappings = data[f"{flag}_mappings"]
        tags = data[f"{flag}_labels"]
        embs = data[f"{flag}_embs"]
        masks = embs['attention_mask']
        embs = embs['input_ids']
        self.data = [(
            torch.flip(embs[m], dims=[0])[:post_num] if len(m) > post_num else torch.flip(embs[m], dims=[0]),
            torch.flip(masks[m], dims=[0])[:post_num] if len(m) > post_num else torch.flip(masks[m], dims=[0]),
            tag
        ) for m, tag in zip(mappings, tags)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
