import importlib

import torch
from lightning import LightningModule
from sklearn.metrics import precision_recall_fscore_support
from torch.nn import functional as F

from model.early_dec_esdm import all_loss_fn, max_action


class MInterface(LightningModule):
    def __init__(
            self,
            model_name='early_dec_han',
            loss='mse',
            bert_lr=1e-6,
            lr=2e-5,
            lr_scheduler=None,
            batch_size=1,
            max_len=64,
            output_size=1,
    ):
        super().__init__()
        self.bert_lr = bert_lr
        self.output_size = output_size
        self.max_len = max_len
        self.batch_size = batch_size
        self.model_name = model_name
        self.lr_scheduler = lr_scheduler
        self.lr = lr
        self.loss = loss
        self.save_hyperparameters()
        self.model = self.load_model()
        self.loss_function = self.configure_loss()
        self.validation_step_outputs = []
        self.validation_step_outputs_early = []
        self.labels = []
        self.max_f1 = 0

    def forward(self, input_ids, attention_mask, lengths):
        if self.model_name == 'early_dec_han':
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lengths=lengths
            )
        else:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

    def training_step(self, batch, batch_idx):
        device = next(self.parameters()).device

        if self.model_name == 'early_dec_han':
            input_ids, attention_mask, lengths, labels = batch
            input_ids = input_ids.view(self.hparams.batch_size * max(lengths), self.hparams.max_len)
            attention_mask = attention_mask.view(self.hparams.batch_size * max(lengths), self.hparams.max_len)
            labels = torch.tensor(labels).to(device)
            output = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lengths=lengths
            )
            loss = self.loss_function(output, labels.float())
        else:
            input_ids, attention_mask, labels = batch
            action_logits, pred_logits = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lengths=None
            )  # batch_size,seq_len,3 1,seq_len,3
            loss = all_loss_fn(action_logits, pred_logits, labels)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        device = next(self.parameters()).device

        if self.model_name == 'early_dec_han':
            input_ids, attention_mask, lengths, labels = batch
            self.labels.extend(labels)
            input_ids = input_ids.view(self.hparams.batch_size * max(lengths), self.hparams.max_len)
            attention_mask = attention_mask.view(self.hparams.batch_size * max(lengths), self.hparams.max_len)
            labels = torch.tensor(labels).to(device)
            output = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lengths=lengths
            )
            loss = self.loss_function(output, labels.float())
            for item in output:
                if item < 0.5:
                    self.validation_step_outputs.append(0)
                else:
                    self.validation_step_outputs.append(1)
        else:
            input_ids, attention_mask, labels = batch
            self.labels.extend(labels.cpu())
            action_logits, pred_logits = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lengths=None
            )  # batch_size,seq_len,3 1,seq_len,3
            loss = all_loss_fn(action_logits, pred_logits, labels)
            action_list = max_action(action_logits)
            choose_logits, final_logits = pred_logits[:, action_list.shape[0] - 1, :], pred_logits[:, -1, :]
            (value, pred), (final_value, final_pred) = torch.max(choose_logits, dim=-1), torch.max(final_logits, dim=-1)
            self.validation_step_outputs_early.append(pred.view(1))
            self.validation_step_outputs.append(final_pred.view(1))
        # self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        precision, recall, f1, support = precision_recall_fscore_support(
            torch.cat(self.validation_step_outputs, dim=0).view(-1).cpu(),
            self.labels,
            zero_division=0.0,
            average='binary'
        )
        self.log('F1_tri', f1, on_step=False, on_epoch=True, prog_bar=True)
        precision, recall, f1, support = precision_recall_fscore_support(
            torch.cat(self.validation_step_outputs_early, dim=0).view(-1).cpu(),
            self.labels,
            zero_division=0.0,
            average='binary'
        )
        self.log('F1_ear', f1, on_step=False, on_epoch=True, prog_bar=True)
        if f1 > self.max_f1:
            self.max_f1 = f1
        self.log('Max-F1', self.max_f1, on_step=False, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.clear()
        self.validation_step_outputs_early.clear()
        self.labels.clear()
        print('')

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        if self.model_name == 'early_dec_han':
            # 创建不同部分的参数组
            bert_params = list(self.model.bert.parameters())
            lstm_params = list(self.model.lstm.parameters())
            attention_params = list(self.model.attention.parameters())
            classifier_params = list(self.model.classifier.parameters())

            # 构建不同参数组
            param_groups = [
                {'params': bert_params, 'lr': self.hparams.bert_lr},
                {'params': lstm_params, 'lr': self.hparams.lr},
                {'params': attention_params, 'lr': self.hparams.lr},
                {'params': classifier_params, 'lr': self.hparams.lr}
            ]
            optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
        else:
            bert_params = list(map(id, list(self.model.bert.parameters())))
            optimizer = torch.optim.AdamW([
                {'params': filter(lambda p: id(p) not in bert_params, self.model.parameters()), 'lr': self.lr},
                {'params': self.model.bert.parameters(), 'lr': self.bert_lr}
            ], weight_decay=1e-3)

        return optimizer

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            return F.mse_loss
        elif loss == 'l1':
            return F.l1_loss
        elif loss == 'bce':
            return F.binary_cross_entropy
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        """
        Change the `snake_case.py` file name to `CamelCase` class name.
        Name model file name as `snake_case.py`
        Name class name `CamelCase`
        :return:
        """
        name = self.hparams.model_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module('.' + name, package=__package__), camel_name)
        except:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        return Model(self.hparams)
