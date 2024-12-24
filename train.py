import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from src.config import TrainingConfig
from src.dataset import ImageNetStreamingDataset, get_transforms
from src.model import ResNet50Module
import torch

def main():
    # Initialize config
    config = TrainingConfig()
    
    # Create datasets
    train_dataset = ImageNetStreamingDataset(
        split="train",
        transform=get_transforms(config, is_train=True)
    )
    
    val_dataset = ImageNetStreamingDataset(
        split="validation",
        transform=get_transforms(config, is_train=False)
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = ResNet50Module(config)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_acc1',
            mode='max',
            save_top_k=1,
            filename='resnet50-{epoch:02d}-{val_acc1:.2f}'
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        precision=config.precision,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        deterministic=False
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main() 