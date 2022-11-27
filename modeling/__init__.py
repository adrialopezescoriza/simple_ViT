# encoding: utf-8
import torch.cuda

from .example_model import ResNet18
from .vit_model import MyViT


def build_model(cfg):
    # model = ResNet18(cfg.MODEL.NUM_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyViT(out_d=cfg.MODEL.NUM_CLASSES).to(device)
    return model
