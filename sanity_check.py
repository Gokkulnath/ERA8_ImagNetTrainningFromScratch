import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from src.config import TrainingConfig
from src.dataset import ImageNetStreamingDataset, get_transforms
from src.model import ResNet50Module
import torch

def main():
    # Initialize config
    config = TrainingConfig()
    
    # Create datasets with limited samples
    train_dataset = ImageNetStreamingDataset(
        split="train",
        transform=get_transforms(config, is_train=True),
        max_samples=100_000  # Limit to 100k samples
    )
    
    val_dataset = ImageNetStreamingDataset(
        split="validation",
        transform=get_transforms(config, is_train=False),
        max_samples=10_000  # 10% of training for validation
    )
    
    # Create dataloaders with DDP compatibility
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True  # Add this for DDP
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True  # Add this for DDP
    )
    
    # Create model
    model = ResNet50Module(config)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_acc1',
            mode='max',
            save_top_k=1,
            filename='resnet50-sanity-{epoch:02d}-{val_acc1:.2f}'
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    
    # Print training configuration
    
    # Initialize trainer with dynamic accelerator config
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        precision=config.precision,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        deterministic=False,
        max_steps=5000,  # Limit steps for quick sanity check
        val_check_interval=250,  # Run validation every 50 training steps
        num_sanity_val_steps=2,  # Run 2 validation batches at the start
        accelerator='auto',  # Let Lightning choose the best accelerator
        devices='auto'  # Let Lightning choose the optimal device setup
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main() 