from typing import List
import torch
from torch import Tensor
from torch import nn
from methods import Optimizer


def update_params(model: nn.Module, u: float, epsilon: List[Tensor]) -> nn.Module:
    """ Update the model parameters with theta = theta + u * epsilon

    Args:
        model: the model to update
        u: the step size
        epsilon: the noise to add to the parameters

    Returns:
        the updated model
    """
    params = model.parameters()

    # Update the model parameters with theta = theta + u * epsilon
    for param, eps in zip(params, epsilon):
        param.data.add_(eps * u)

    return model


def zo_gradient(model: nn.Module, criterion, u: float, epsilon: List[Tensor], batch_x: Tensor, batch_y: Tensor):
    """ Compute the zeroth order gradient of the loss function

    Args:
        model: the model to update
        loss_fn: the loss function
        u: the step size
        epsilon: the noise to add to the parameters
        x: the input data
        y: the target data

    Returns:
        the zeroth order gradient of the loss function
    """
    # Model at theta + u * epsilon
    update_params(model, u, epsilon)
    l_updated = criterion(model(batch_x), batch_y)

    # Model at theta - u * epsilon
    update_params(model, -2*u, epsilon)
    delta = (l_updated - criterion(model(batch_x), batch_y)) / (2*u)

    # Model at theta
    update_params(model, u, epsilon)

    gradients = [eps * delta for eps in epsilon]
    return gradients


class ZeroOrderOptimizer(Optimizer):
    """ Zeroth order optimizer """

    def __init__(self, u: float, eta: float):
        self.u = u
        self.eta = eta

    @torch.no_grad()
    def update_model(self, model, batch, criterion, metrics, device):
        batch_x, batch_y = batch
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Compute gradients for the batch
        epsilon = [torch.randn_like(param) for param in model.parameters()]
        epsilon_norm = torch.norm(torch.cat([eps.flatten() for eps in epsilon]), 2)
        epsilon = [eps / epsilon_norm for eps in epsilon]
        gradients = zo_gradient(model, criterion, self.u, epsilon, batch_x, batch_y)

        # Update the model parameters
        update_params(model, -self.eta, gradients)

        # Compute the loss and metrics
        prediction = model(batch_x)
        loss = criterion(prediction, batch_y)
        metrics_result = {name: met(prediction, batch_y) for name, met in metrics.items()}

        return loss.item(), metrics_result
