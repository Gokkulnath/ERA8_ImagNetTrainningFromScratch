from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Training params
    batch_size: int = 64
    num_workers: int = 4
    max_epochs: int = 2
    # batch_size: int = 1024
    # num_workers: int = 8
    # max_epochs: int = 100
    precision: str = "16-mixed"  # Use mixed precision training
    
    # Optimizer params
    base_lr: float = 0.1 * (batch_size / 256)  # Scale learning rate with batch size
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Data params
    image_size: int = 224
    num_classes: int = 1000
    
    # Training techniques
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Scheduler params
    warmup_epochs: int = 5
    min_lr: float = 1e-5 