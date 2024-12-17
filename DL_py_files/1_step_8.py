import torch.nn as nn
import torch


def function04(x: torch.Tensor, y: torch.Tensor) -> nn.Linear:
    layer = nn.Linear(in_features=x.shape[1], out_features=1, bias=True)

    max_epochs = 1000
    target_loss = 0.3
    step = 1e-2

    for epoch in range(max_epochs):

        y_pred = layer(x).ravel()

        loss = torch.mean((y_pred - y) ** 2)

        if loss < target_loss:
            break

        loss.backward()

        with torch.no_grad():
            layer.weight -= layer.weight.grad * step
            layer.bias -= layer.bias.grad * step

        layer.zero_grad()

    return layer