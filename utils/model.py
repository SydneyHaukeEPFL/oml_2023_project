import torch
import torchvision
import torch.backends.cudnn
from torch import nn


class MLP(nn.Module):
    """ A simple MLP """
    def __init__(self, input_shape, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_shape[0]*input_shape[1]*input_shape[2], hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.log_softmax(x, dim=1)
        return x


class SimpleCNN(nn.Module):
    """ A simple CNN """
    def __init__(self, input_shape, output_size) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.fc1 = nn.Linear(input_shape[1] * input_shape[2] * 32, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.log_softmax(self.fc2(x), dim=1)
        return x


def get_model(config, device):
    """
    :param device: instance of torch.device
    :return: An instance of torch.nn.Module
    """
    num_classes = config["num_classes"]

    model = {
        "mlp": lambda: MLP(config["input_shape"], 128, num_classes),
        "simple_cnn": lambda: SimpleCNN(config["input_shape"], num_classes),
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
