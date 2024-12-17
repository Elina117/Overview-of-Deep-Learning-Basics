import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    predictions = []
    model.eval()

    for (x, y) in loader:
        output = model(x)
        predict_classes = output.argmax(dim=1)
        predictions.append(predict_classes)

    return torch.cat(predictions)