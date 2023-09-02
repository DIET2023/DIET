"""
Created by yizhi.chen.
"""

import cv2
import torch
import torchvision.models as models
from torch import nn
import pytorch_lightning as pl
#import segmentation_models_pytorch as smp

#from common.loss import MixLoss, DiceLoss
#from common.unet import UNet
#from common.scheduler import StepLRwithWarmUp
from .loss import MixLoss, DiceLoss
from .unet import UNet
from .scheduler import StepLRwithWarmUp


class ClassificationModel(pl.LightningModule):
    def __init__(self, num_classes=2):
        """
        Args:
        """
        super().__init__()
        self.cnn_model = models.resnet50(pretrained=True)
        self.cnn_model.fc = nn.Linear(self.cnn_model.fc.weight.shape[1], num_classes)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.cnn_model(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        prediction = self.cnn_model(x)
        loss = self.loss_func(prediction, y)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.cnn_model(x)
        loss = self.loss_func(prediction, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        # https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html#learning-rate-scheduling

        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4, )
        scheduler = {
            "scheduler": StepLRwithWarmUp(optimizer, 1000, gamma=0.9, warmup_step=5000),
            "interval": "step",
        }
        return [optimizer], [scheduler]



