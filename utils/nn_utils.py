from itertools import zip_longest
from typing import Union, Iterable

import torch
from torch import nn
from torch.nn import Linear, Sequential
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from utils.utility_classes import NormalizerLoss

def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def hard_update(src_net: nn.Module, dst_net: nn.Module):
    with torch.no_grad():
        for target_param, param in zip_strict(dst_net.parameters(), src_net.parameters()):
            target_param.data.copy_(param.data)


def soft_update(src_net: nn.Module, dst_net: nn.Module, tau: float = 0.5) -> None:
    """
    Perform a Polyak average update on ``dst_net`` using ``src_net``:
    :param src_net: the network whose parameters are used to update the target params
    :param dst_net: the network whose parameters are to update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with torch.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip_strict(dst_net.parameters(), src_net.parameters()):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
    """
    COPIED FROM sb3.
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'mps', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        # elif torch.backends.mps.is_available():
        #     device = "mps"
        else:
            device = "cpu"

    device = torch.device(device)

    return device


def get_optimizer(model: nn.Module, type_: str, lr: float):
    if type_ == 'sgd':
        optim = torch.optim.SGD(params=model.parameters(), lr=lr)
    elif type_ == 'adam':
        optim = torch.optim.Adam(params=model.parameters(), lr=lr)
    elif type_ == 'adamw':
        optim = torch.optim.AdamW(params=model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f'{type_} optimizer not implemented yet')
    return optim


def get_scheduler(optim: torch.optim.lr_scheduler, type_: str, num_epochs: int):
    if type_ is None:
        return None
    if type_ == 'exp':
        scheduler = ExponentialLR(optimizer=optim, gamma=0.99)
    elif type_ == 'step':
        scheduler = StepLR(optim, int(0.3 * num_epochs))
    else:
        raise NotImplementedError(f'{type_} scheduler not implemented yet')
    # ADD HERE NEW SCHEDULERS
    return scheduler


def get_loss_func(type_: str, normalized=True, **kwargs):
    if normalized:
        loss_func = NormalizerLoss(loss_type=type_, **kwargs)
    else:
        if type_ == 'mse':
            loss_func = nn.MSELoss()
        elif type_ == 'cross_ent':
            loss_func = nn.CrossEntropyLoss()
        elif type_ == 'l1':
            loss_func = nn.L1Loss()
        else:
            raise NotImplementedError(f'{type_} loss function not implemented yet')
    return loss_func


def get_test_metric(type_: str):
    if type_ == 'l1':
        metric = torch.nn.functional.l1_loss
    else:
        raise NotImplementedError(f'{type_} metric function not implemented yet')
    return metric


def cranmer_init(sequence: Sequential, hidden_dim):
    """
    Custom weight initialization for LNN proposed by Cranmer et al.,
    Lagrangian Neural Networks,
    arXiv. https://doi.org/10.48550/arXiv.2003.04630,
    2020
    """
    num_linear = len(sequence) // 2 + 1
    for i, layer in enumerate(sequence):
        if isinstance(layer, Linear):
            if i == 0:
                sigma = 2.2 / hidden_dim**0.5
            elif 0 < i < num_linear:
                sigma = 0.58 * i / hidden_dim**0.5
            else:
                sigma = hidden_dim**0.5
            torch.nn.init.normal_(layer.weight, mean=0, std=sigma)
            torch.nn.init.constant_(layer.bias, 0)


def actor_critic_params(actor, critic):
    nets = [actor, critic]
    for net in nets:
        for param in net.parameters():
            yield param


def standardize_batch(data: torch.Tensor) -> torch.Tensor:
    """
    This function performs the Z-Score Normalization, i.e. subtracts from each column of the (2D)array of data its mean,
    divides the result by its standard deviation and returns the transformed tensor
    Args:
        data: the tensor to transform

    Returns: the standardized tensor
    """
    if type(data) is not torch.Tensor:
        data = torch.tensor(data)
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)

    assert torch.all(std != 0)

    return (data - mean) / std
