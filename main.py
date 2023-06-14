import argparse
import os
import torch
from torchmetrics import Accuracy
from utils.training import train, get_optimizer
from utils.model import get_model
from utils.dataset import get_dataset


def get_args():
    """ Get arguments from command line """
    parser = argparse.ArgumentParser(description="Training")
    # Dataset
    parser.add_argument("--dataset", type=str, default="Cifar10", help="dataset", choices=["Cifar10", "Cifar100", "FashionMNIST"])

    # Model
    parser.add_argument("--model", type=str, default="resnet18", help="model", choices=["mlp", "simple_cnn", "resnet18", "resnet50", "resnet101", "vgg11", "vgg11_bn"])
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", help="optimizer", choices=["zero_order", "sgd", "adam"])
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    # # SGD
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    # # Zero-order
    parser.add_argument("--u", type=float, default=0.01, help="u")

    return parser.parse_args()


def main():
    """ Main """
    config = vars(get_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dl, test_dl, criterion = get_dataset(config, num_workers=4)
    model = get_model(config, device)
    optimizer = get_optimizer(config, model.parameters())
    metrics = {
        "Accuracy": Accuracy(task="multiclass", num_classes=config["num_classes"]).to(device),
    }

    results_file = f"results/{config['dataset']}_{config['model']}_{config['optimizer']}_{len(os.listdir('results'))}.txt"

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Config:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        f.write("\n")
        f.write("Epoch:\n")

    for epoch, results in enumerate(train(model, optimizer, train_dl, test_dl, criterion, config["epochs"], device, metrics)):
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(f"  {epoch}:\n")
            for key, value in results.items():
                f.write(f"    {key}: {value}\n")


if __name__ == "__main__":
    main()
