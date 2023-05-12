import torch
import argparse
from training import train, get_optimizer
from utils.model import get_model
from utils.dataset import get_dataset


def get_args():
    parser = argparse.ArgumentParser(description="Training")
    # Dataset
    parser.add_argument("--dataset", type=str, default="Cifar10", help="dataset", choices=["Cifar10", "Cifar100", "FashionMNIST"])

    # Model
    parser.add_argument("--model", type=str, default="resnet18", help="model", choices=["mlp", "simple_cnn", "resnet18", "resnet50", "resnet101" "vgg11", "vgg11_bn"])
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", help="optimizer", choices=["zero_order", "sgd", "adam"])
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    # # Adam
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    # # Zero-order
    parser.add_argument("--u", type=float, default=0.01, help="u")

    return parser.parse_args()


def main():
    config = vars(get_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    train_loader, test_loader = get_dataset(config, num_workers=4)
    model = get_model(config, device)
    optimizer = get_optimizer(config, model.parameters())

    train(model, optimizer, train_loader, test_loader, config["epochs"], device)


if __name__ == "__main__":
    main()
