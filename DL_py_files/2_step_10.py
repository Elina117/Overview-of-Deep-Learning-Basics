import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for inputs, target in data_loader:
        output = model(inputs)
        loss = loss_fn(output, target)

        total_loss += loss.item()
        num_batches += 1

    mean_loss = total_loss / num_batches
    return mean_loss