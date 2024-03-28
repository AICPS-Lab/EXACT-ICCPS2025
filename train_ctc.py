# https://discuss.pytorch.org/t/proper-way-to-use-torch-nn-ctcloss/28909

# proper way to train this and might have to set up a different loader.
import math
import os
from methods import Segmenter, TransformerModel, LSTM, CRNN, UNet,CCRNN, UNet2, PatchTST
from matplotlib import pyplot as plt
import torch
from methods.unetc import UNetc
from methods.unetr import UNetrt
from utilities import printc, seed
from utils_loader import get_dataloaders, test_idea_dataloader_ABC_to_BCA, test_idea_dataloader_long_A_B_to_AB
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils_metrics import eval_dense_label_to_classification, mean_iou, visualize_softmax
from torch.nn import functional as F


def main(config):
    
    seed(config['seed'])