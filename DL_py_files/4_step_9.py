import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def predict_tta(model: nn.Module, loader: DataLoader, device: torch.device, iterations: int = 2):
    model.eval()
    all_logits = []

    for i in range(iterations):
        logits_iter = []

        for images, _ in loader:  # так как возвращает кортеж изображения и метки
            output = model(images)
            logits_iter.append(output)

        logits_iter = torch.cat(logits_iter, dim=0)
        all_logits.append(logits_iter)

    all_logits = torch.stack(all_logits, dim=-1)  # Размерность [N, C, iterations]
    avg_logits = all_logits.mean(dim=-1)  # Размерность [N, C]

    predictions_classes = avg_logits.argmax(dim=1)  # Выбираем класс с максимальной вероятностью

    return predictions_classes
