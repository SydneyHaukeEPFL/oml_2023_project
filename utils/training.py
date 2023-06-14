import torch
from torch.utils.data import DataLoader
from utils import accumulators
from methods import Optimizer
from methods.one_order_optim import TorchOptimizer
from methods.zero_order_optim import ZeroOrderOptimizer


def get_optimizer(config, model_parameters):
    """ Get the optimizer from the configuration """
    if config["optimizer"].lower() == "sgd":
        optimizer = TorchOptimizer(torch.optim.SGD(model_parameters, lr=config["lr"], momentum=config["momentum"]))
    elif config["optimizer"].lower() == "adam":
        optimizer = TorchOptimizer(torch.optim.Adam(model_parameters, lr=config["lr"]))
    elif config["optimizer"].lower() == "zero_order":
        optimizer = ZeroOrderOptimizer(config["u"], eta=config["lr"])
    else:
        raise ValueError(f"Unknown optimizer {config['optimizer']}")
    return optimizer


def train_epoch(
    model: torch.nn.Module,
    optimizer: Optimizer,
    train_dl: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    metrics: dict
):
    """
    Train a model for one epoch

    Args:
        model: model to train
        optimizer: optimizer to use
        train_dl: training data loader
        criterion: loss function
        num_epochs: number of epochs
        device: device to use for computation
        metrics: a dict of metrics to keep track of
    """

    # Enable training mode (automatic differentiation + batch norm)
    model.train()

    # Keep track of statistics during training
    for metric in metrics.values():
        metric.reset()
    mean_train_loss = accumulators.Mean()

    for batch_x, batch_y in train_dl:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        loss, _ = optimizer.update_model(model, (batch_x, batch_y), criterion, metrics, device)

        # Store the statistics
        mean_train_loss.add(loss, weight=len(batch_x))

    return mean_train_loss.value(), metrics


def test(
    model: torch.nn.Module,
    test_dl: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    metrics: dict
):
    """
    Test a model
    """
    # Evaluation
    model.eval()

    # Keep track of statistics during training
    for metric in metrics.values():
        metric.reset()
    mean_train_loss = accumulators.Mean()

    with torch.no_grad():
        for batch_x, batch_y in test_dl:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)

            # Store the statistic
            mean_train_loss.add(loss, weight=len(batch_x))
            for met in metrics.values():
                met(prediction, batch_y)

    return mean_train_loss.value(), metrics


def train(
    model: torch.nn.Module,
    optimizer: Optimizer,
    train_dl: DataLoader,
    test_dl: DataLoader,
    criterion: torch.nn.Module,
    num_epochs: int,
    device: torch.device,
    metrics: dict
):
    """ Train a model """
    for epoch in range(num_epochs):
        loss, metrics = train_epoch(model, optimizer, train_dl, criterion, device, metrics)
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"   Train loss: {loss:.4f}")
        for name, met in metrics.items():
            print(f"   {name}: {met.compute():.4f}")
        res = {"loss": loss}
        for name, met in metrics.items():
            res[name] = met.compute()

        loss, metrics = test(model, test_dl, criterion, device, metrics)
        print(f"   Val loss: {loss:.4f}")
        for name, met in metrics.items():
            print(f"   Val {name}: {met.compute():.4f}")
        print()
        res["Val loss"] = loss
        for name, met in metrics.items():
            res[f"Val {name}"] = met.compute()
        yield res
