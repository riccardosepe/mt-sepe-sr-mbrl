import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0, last_epoch=-1, verbose=False):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = float(self.last_epoch) / float(self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            t = self.last_epoch - self.warmup_epochs
            T = self.max_epochs - self.warmup_epochs
            alpha = 0.5 * (1.0 + np.cos(np.pi * t / T))
            return [(1 - alpha) * self.eta_min + alpha * base_lr for base_lr in self.base_lrs]


if __name__ == '__main__':
    def cosine_decay_lr(initial_lr, current_epoch, total_epochs):
        """
        Compute the learning rate for a single cosine decay schedule.

        Args:
            initial_lr (float): Initial learning rate.
            current_epoch (int): Current epoch number (starting from 1).
            total_epochs (int): Total number of epochs.

        Returns:
            float: Learning rate for the current epoch.
        """
        alpha = 0.5 * (1 + np.cos(np.pi * current_epoch / total_epochs))
        decayed_lr = (alpha-1) * initial_lr
        return decayed_lr

    # Example usage
    initial_lr = 0.1
    total_epochs = 50

    lrs = [initial_lr]

    for epoch in range(1, total_epochs):
        lr = cosine_decay_lr(initial_lr, epoch, total_epochs)
        lrs.append(lr)

    plt.plot(np.arange(total_epochs), lrs)
    plt.show()
    print(lrs)
