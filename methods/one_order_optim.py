from methods import Optimizer
import torch


class TorchOptimizer(Optimizer):
    """ Wrapper around torch.optim.Optimizer """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def update_model(self, model, batch, criterion, metrics, device):
        batch_x, batch_y = batch
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Compute gradients for the batch
        self.optimizer.zero_grad()
        prediction = model(batch_x)
        loss = criterion(prediction, batch_y)
        metrics_result = {name: met(prediction, batch_y) for name, met in metrics.items()}
        loss.backward()

        # Do an optimizer step
        self.optimizer.step()

        return loss.item(), metrics_result
