import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, L1Loss


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


# function to convert a value of seconds to a string of format "minutes: mm, seconds: ss.milliseconds"
def sec_to_min(seconds):
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes}min, {seconds:.2f}s"


class ConstrainedRegressionLoss(Module):
    """
    Loss function for regression with a penalty for values above a certain threshold
    """
    def __init__(self,
                 bounds: Tensor,
                 method: str = 'l1',
                 n: int = None,
                 penalty: float = 1.):
        """
        Args:
            method: the method to use for the basic loss. Currently only 'l1' is supported
            n: the number of features to apply the penalty to. If None, the penalty is applied to all the features
            bounds: the bounds for the penalty. It must be a tensor of shape (n,)
            penalty: The coefficient of the applied penalty. If it is 0, the loss is equivalent to the basic loss
        """
        super(ConstrainedRegressionLoss, self).__init__()
        self.penalty = penalty
        if method == 'l1':
            self.fn = F.l1_loss
        else:
            raise ValueError(f"Method {method} not recognized")
        if type(bounds) is not Tensor:
            raise ValueError(f"Type {type(bounds)} not valid for bounds")

        self.bounds = bounds.squeeze()

        if n is None:
            self.n = bounds.shape[0]
        else:
            self.n = n

        assert self.n == self.bounds.shape[0], \
            f"Number of features {self.n} and bounds {self.bounds.shape[0]} don't match"

    def forward(self, pred: Tensor, target: Tensor):
        basic_loss = self.fn(pred, target)
        indices = torch.any(pred[:, :self.n].abs() > self.bounds, dim=1)
        if indices.sum() == 0:
            penalty = 0
        else:
            # tensor_for_penalty = torch.pow(pred[indices, :], 2)


            # penalty = self.penalty * tensor_for_penalty.mean()
            penalty = self.penalty * F.mse_loss(pred[indices, :], target[indices, :])

        return basic_loss + penalty


if __name__ == '__main__':
    bounds = torch.tensor([3, 1, 1])
    bb = bounds.unsqueeze(0)
    B = 64
    qt = torch.rand((B, 3)) * 2 * bb - bb
    bb = torch.tensor([[20, 6, 6]])
    qdt = torch.rand((B, 3)) * 2 * bb - bb
    target = torch.cat((qt, qdt), dim=1)
    bb = torch.tensor([[6, 2, 2]])
    qp = torch.rand((B, 3)) * 2 * bb - bb
    bb = torch.tensor([[40, 12, 12]])
    qdp = torch.rand((B, 3)) * 2 * bb - bb
    pred = torch.cat((qp, qdp), dim=1)
    loss1_fn = L1Loss()
    loss2_fn = ConstrainedRegressionLoss(bounds, n=3, penalty=1.)
    print(loss1_fn(pred, target))
    print(loss2_fn(pred, target))

    bb = torch.tensor([[3, 1, 1]])
    qp = torch.rand((B, 3)) * 2 * bb - bb
    bb = torch.tensor([[20, 6, 6]])
    qdp = torch.rand((B, 3)) * 2 * bb - bb
    pred = torch.cat((qp, qdp), dim=1)
    print(loss1_fn(pred, target))
    print(loss2_fn(pred, target))
