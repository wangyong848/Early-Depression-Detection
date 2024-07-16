# main.py
import torch
from lightning.pytorch.cli import LightningCLI

from data import DInterface
from model import MInterface


def main():
    torch.set_float32_matmul_precision('high')
    LightningCLI(MInterface, DInterface)


if __name__ == "__main__":
    main()
