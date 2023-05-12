import os
import torch
import torch.utils.data
import torch.backends.cudnn
import torchvision


def get_CIFAR(
    config,
    test_batch_size=1000,
    shuffle_train=True,
    num_workers=2,
    data_root=os.getenv("DATA_DIR", "./data"),
):
    """
    Create dataset loaders for the chosen dataset
    :return: Tuple (training_loader, test_loader)
    """
    config["input_shape"] = [3, 32, 32]

    if config["dataset"] == "Cifar10":
        dataset = torchvision.datasets.CIFAR10
        config["num_classes"] = 10
    elif config["dataset"] == "Cifar100":
        dataset = torchvision.datasets.CIFAR100
        config["num_classes"] = 100
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


def get_FashionMNIST(
    config,
    test_batch_size=1000,
    shuffle_train=True,
    num_workers=2,
    data_root=os.getenv("DATA_DIR", "./data"),
):
    """
    Create dataset loaders for the chosen dataset
    :return: Tuple (training_loader, test_loader)
    """
    config["input_shape"] = [1, 32, 32]

    dataset = torchvision.datasets.FashionMNIST

    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(data_mean, data_stddev),
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


def get_dataset(
    config,
    test_batch_size=1000,
    shuffle_train=True,
    num_workers=2,
    data_root=os.getenv("DATA_DIR", "./data"),
):
    """
    Create dataset loaders for the chosen dataset
    :return: Tuple (training_loader, test_loader)
    """

    if config["dataset"] in ["Cifar10", "Cifar100"]:
        return get_CIFAR(
            config,
            test_batch_size,
            shuffle_train,
            num_workers,
            data_root,
        )
    elif config["dataset"] == "FashionMNIST":
        return get_FashionMNIST(
            config,
            test_batch_size,
            shuffle_train,
            num_workers,
            data_root,
        )
    else:
        raise ValueError(
            "Unexpected value for config[dataset] {}".format(config["dataset"])
        )
