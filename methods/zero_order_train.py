#!/usr/bin/env python3

import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from typing import List
import torchvision
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import utils.accumulators


config = dict(
    dataset="Cifar10",
    model="resnet18",

    batch_size=256,
    num_epochs=300,
    seed=42,
)

hparams = dict(
    u=0.001,
    eta=100,
)


output_dir = "./output.tmp"  # Can be overwritten by a script calling this
writer = SummaryWriter(output_dir)
writer.add_hparams(config, {})


def loss(model_updated, loss_fn, x, y):
    return loss_fn(model_updated(x), y)


def update_model(model: nn.Module, u: float, epsilon: List[Tensor]) -> nn.Module:
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


def zo_gradient(model: nn.Module, loss_fn, u: float, epsilon: List[Tensor], x: Tensor, y: Tensor):
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
    update_model(model, u, epsilon)
    l_updated = loss(model, loss_fn, x, y)

    # Model at theta - u * epsilon
    update_model(model, -2*u, epsilon)
    delta = (l_updated - loss(model, loss_fn, x, y)) / (2*u)

    # Model at theta
    update_model(model, u, epsilon)

    gradients = [eps * delta for eps in epsilon]
    return gradients


def zo_update(model: nn.Module, loss_fn, u: float, epsilon: List[Tensor], eta: float, x: Tensor, y: Tensor):
    gradients = zo_gradient(model, loss_fn, u, epsilon, x, y)

    # TODO : return or set model with new parameters
    # new_parameters = model.parameters() - eta * gradient
    # Model at theta - eta * gradients
    update_model(model, -eta, gradients)
    return


def step(model, batch)


def main():
    """
    Train a model
    You can either call this script directly (using the default parameters),
    or import it as a module, override config and run main()
    :return: scalar of the best accuracy
    """

    # Set the seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # We will run on CUDA if there is a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configure the dataset, model and the optimizer based on the global
    # `config` dictionary.
    training_loader, test_loader = get_dataset()
    model = get_model(device)
    criterion = torch.nn.CrossEntropyLoss()

    # We keep track of the best accuracy so far to store checkpoints
    best_accuracy_so_far = utils.accumulators.Max()

    for epoch in range(config["num_epochs"]):
        print("Epoch {:03d}".format(epoch))

        # Enable training mode (automatic differentiation + batch norm)
        model.train()

        # Keep track of statistics during training
        mean_train_accuracy = utils.accumulators.Mean()
        mean_train_loss = utils.accumulators.Mean()

        with torch.no_grad():
            for batch_x, batch_y in training_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # Optimization step
                epsilon = [torch.randn_like(param) for param in model.parameters()]
                epsilon_norm = torch.norm(torch.cat([eps.flatten() for eps in epsilon]), 2)
                epsilon = [eps / epsilon_norm for eps in epsilon]
                zo_update(model, criterion, config["u"], epsilon, config["eta"], batch_x, batch_y)

                # Metrics
                prediction = model(batch_x)
                loss = criterion(prediction, batch_y)
                acc = accuracy(prediction, batch_y)

            # Store the statistics
            mean_train_loss.add(loss.item(), weight=len(batch_x))
            mean_train_accuracy.add(acc.item(), weight=len(batch_x))

        # Log training stats
        log_metric(
            "accuracy",
            {"epoch": epoch, "value": mean_train_accuracy.value()},
            {"split": "train"},
        )
        log_metric(
            "cross_entropy",
            {"epoch": epoch, "value": mean_train_loss.value()},
            {"split": "train"},
        )

        # Evaluation
        model.eval()
        mean_test_accuracy = utils.accumulators.Mean()
        mean_test_loss = utils.accumulators.Mean()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                prediction = model(batch_x)
                loss = criterion(prediction, batch_y)
                acc = accuracy(prediction, batch_y)
                mean_test_loss.add(loss.item(), weight=len(batch_x))
                mean_test_accuracy.add(acc.item(), weight=len(batch_x))

        # Log test stats
        log_metric(
            "accuracy",
            {"epoch": epoch, "value": mean_test_accuracy.value()},
            {"split": "test"},
        )
        log_metric(
            "cross_entropy",
            {"epoch": epoch, "value": mean_test_loss.value()},
            {"split": "test"},
        )

        best_accuracy_so_far.add(mean_test_accuracy.value())

        # Log for tensorboard
        writer.add_scalar('Loss/train', mean_train_loss.value(), epoch)
        writer.add_scalar('Accuracy/train', mean_train_accuracy.value(), epoch)

    # Return the optimal accuracy, could be used for learning rate tuning
    return best_accuracy_so_far.value()


def accuracy(predicted_logits, reference):
    """Compute the ratio of correctly predicted labels"""
    labels = torch.argmax(predicted_logits, 1)
    correct_predictions = labels.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()


def log_metric(name, values, tags):
    """
    Log timeseries data.
    Placeholder implementation.
    This function should be overwritten by any script that runs this as a module.
    """
    print("{name}: {values} ({tags})".format(name=name, values=values, tags=tags))


def get_dataset(
    test_batch_size=1000,
    shuffle_train=True,
    num_workers=2,
    data_root=os.getenv("DATA_DIR", "./data"),
):
    """
    Create dataset loaders for the chosen dataset
    :return: Tuple (training_loader, test_loader)
    """
    if config["dataset"] == "Cifar10":
        dataset = torchvision.datasets.CIFAR10
    elif config["dataset"] == "Cifar100":
        dataset = torchvision.datasets.CIFAR100
    else:
        raise ValueError(
            "Unexpected value for config[dataset] {}".format(config["dataset"])
        )

    data_mean = (0.4914, 0.4822, 0.4465)
    data_stddev = (0.2023, 0.1994, 0.2010)

    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )

    training_set = dataset(
        root=data_root, train=True, download=True, transform=transform_train
    )
    test_set = dataset(
        root=data_root, train=False, download=True, transform=transform_test
    )

    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=config["batch_size"],
        shuffle=shuffle_train,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )

    return training_loader, test_loader


def get_model(device):
    """
    :param device: instance of torch.device
    :return: An instance of torch.nn.Module
    """
    num_classes = 100 if config["dataset"] == "Cifar100" else 10

    model = {
        "vgg11": lambda: torchvision.models.vgg11(num_classes=num_classes),
        "vgg11_bn": lambda: torchvision.models.vgg11_bn(num_classes=num_classes),
        "resnet18": lambda: torchvision.models.resnet18(num_classes=num_classes),
        "resnet50": lambda: torchvision.models.resnet50(num_classes=num_classes),
        "resnet101": lambda: torchvision.models.resnet101(num_classes=num_classes),
    }[config["model"]]()

    model.to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    return model


if __name__ == "__main__":
    main()
