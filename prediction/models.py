import pytorch_lightning as pl
from torchvision import models
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from torchmetrics import Accuracy,AUROC
from torchmetrics.classification import MultilabelAUROC


class ResNet(pl.LightningModule):
    def __init__(self, num_classes,lr,pretrained):
        super().__init__()
        self.model_name = 'resnet'
        self.num_classes = num_classes
        self.pretrained=pretrained
        self.model = models.resnet34(pretrained=self.pretrained)
        # freeze_model(self.model)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

        self.lr=lr
        self.accu_func= Accuracy(task="multilabel", num_labels=num_classes)
        self.auroc_func = MultilabelAUROC(num_labels=num_classes,average='macro', thresholds=None)

    def remove_head(self):
        num_features = self.model.fc.in_features
        id_layer = nn.Identity(num_features)
        self.model.fc = id_layer

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=self.lr)
        return optimizer

    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)

        multi_accu = self.accu_func(prob, lab)
        multi_auroc = self.auroc_func(prob,lab.long())
        return loss,multi_accu,multi_auroc

    def training_step(self, batch, batch_idx):
        loss,multi_accu,multi_auroc = self.process_batch(batch)
        self.log('train_loss', loss)
        self.log('train_accu', multi_accu)
        self.log('train_auroc', multi_auroc)
        # grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        # self.logger.experiment.add_image('images', grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, multi_accu, multi_auroc = self.process_batch(batch)
        self.log('val_loss', loss)
        self.log('val_accu', multi_accu)
        self.log('val_auroc', multi_auroc)

    def test_step(self, batch, batch_idx):
        loss,multi_accu,multi_auroc = self.process_batch(batch)
        self.log('test_loss', loss)
        self.log('test_accu', multi_accu)
        self.log('test_auroc', multi_auroc)


class DenseNet(pl.LightningModule):
    def __init__(self, num_classes,lr,pretrained):
        super().__init__()
        self.model_name = 'densenet'
        self.num_classes = num_classes
        self.model = models.densenet121(pretrained=pretrained)
        # freeze_model(self.model)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, self.num_classes)
        self.lr = lr
        self.pretrained = pretrained
        self.accu_func = Accuracy(task="multilabel", num_labels=num_classes)
        self.auroc_func = MultilabelAUROC(num_labels=num_classes, average='macro', thresholds=None)

    def remove_head(self):
        num_features = self.model.classifier.in_features
        id_layer = nn.Identity(num_features)
        self.model.classifier = id_layer

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=self.lr)
        return optimizer

    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)

        multi_accu = self.accu_func(prob, lab)
        multi_auroc = self.auroc_func(prob, lab.long())
        return loss, multi_accu, multi_auroc

    def training_step(self, batch, batch_idx):
        loss, multi_accu, multi_auroc = self.process_batch(batch)
        self.log('train_loss', loss)
        self.log('train_accu', multi_accu)
        self.log('train_auroc', multi_auroc)
        # grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        # self.logger.experiment.add_image('images', grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, multi_accu, multi_auroc = self.process_batch(batch)
        self.log('val_loss', loss)
        self.log('val_accu', multi_accu)
        self.log('val_auroc', multi_auroc)

    def test_step(self, batch, batch_idx):
        loss, multi_accu, multi_auroc = self.process_batch(batch)
        self.log('test_loss', loss)
        self.log('test_accu', multi_accu)
        self.log('test_auroc', multi_auroc)

    def test_step_end(self, output_results):
        prob, lab = output_results
        loss = F.binary_cross_entropy(prob, lab)

        multi_accu = self.accu_func(prob, lab)
        multi_auroc = self.auroc_func(prob, lab.long())
        self.log('test_loss', loss,epoch_end=True)
        self.log('test_accu', multi_accu,epoch_end=True)
        self.log('test_auroc', multi_auroc,epoch_end=True)
