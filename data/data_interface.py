import importlib

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.erisk_data import Collate


class DInterface(LightningDataModule):

    def __init__(
            self,
            num_workers=8,
            dataset='erisk_data',
            max_len=64,
            model_name='early_dec_han'
    ):
        super().__init__()
        self.model_name = model_name
        self.max_len = max_len
        self.data_module = None
        self.test_set = None
        self.val_set = None
        self.train_set = None
        self.num_workers = num_workers
        self.dataset = dataset
        self.batch_size = 1
        self.load_data_module()
        self.save_hyperparameters()

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_set = self.data_module(flag='train', hparams=self.hparams)
            self.val_set = self.data_module(flag='test', hparams=self.hparams)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_set = self.data_module(flag='test', hparams=self.hparams)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            collate_fn=Collate() if self.model_name=='early_dec_han' else None,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            collate_fn=Collate() if self.model_name=='early_dec_han' else None,
            num_workers=self.hparams.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            collate_fn=Collate() if self.model_name=='early_dec_han' else None,
            num_workers=self.hparams.num_workers
        )

    def load_data_module(self):
        """
        Change the `snake_case.py` file name to `CamelCase` class name.
        Name model file name as `snake_case.py`
        Name class name `CamelCase`
        :return:
        """
        name = self.dataset
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module('.' + name, package=__package__), camel_name)
        except:
            raise ValueError(f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')
