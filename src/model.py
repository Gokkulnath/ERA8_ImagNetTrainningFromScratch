import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import SGD
import timm.data.mixup as mixup

class ResNet50Module(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = resnet50(num_classes=config.num_classes)
        
        # Initialize mixup
        self.mixup_fn = mixup.Mixup(
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
            label_smoothing=config.label_smoothing,
            num_classes=config.num_classes
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        
        # Apply mixup/cutmix
        images, targets = self.mixup_fn(images, targets)
        
        # Forward pass
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, targets)
        
        # Calculate accuracy
        acc1, acc5 = self._accuracy(outputs, targets, topk=(1, 5))
        
        # Log metrics - add sync_dist=True for proper multi-GPU logging
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc1', acc1, prog_bar=True, sync_dist=True)
        self.log('val_acc5', acc5, prog_bar=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        # Create optimizer
        optimizer = SGD(
            self.parameters(),
            lr=self.config.base_lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        # Create scheduler
        scheduler = {
            'scheduler': CosineAnnealingLR(
                optimizer,
                T_max=self.config.max_epochs,
                eta_min=self.config.min_lr
            ),
            'interval': 'epoch',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]

    @staticmethod
    def _accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res 