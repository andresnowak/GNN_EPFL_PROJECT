from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_ratio, total_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.total_steps = total_steps
        self.min_lr = 1e-7  # Add minimum learning rate parameter
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            lrs = [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lrs = [base_lr * max(0.0, 1.0 - progress) for base_lr in self.base_lrs]

        # Apply minimum learning rate limit
        lrs = [max(lr, self.min_lr) for lr in lrs]
        return lrs