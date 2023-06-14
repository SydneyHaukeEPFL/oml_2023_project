""" This module contains the base class for optimizers """
from typing import Tuple
import torch
from torch import Tensor


class Optimizer:
    """ Base class for optimizers """

    def update_model(
        self,
        model: torch.nn.Module,
        batch: Tuple[Tensor, Tensor],
        criterion: torch.nn.Module,
        metrics: dict,
        device: torch.device,
    ):
        """ Update the model parameters using the given batch

        Args:
            model: the model to update
            batch: a tuple (batch_x, batch_y) of input and target tensors
            criterion: the loss function
            metrics: a dict of metrics to keep track of
            device: the device to use for the computation
        """
        raise NotImplementedError()
