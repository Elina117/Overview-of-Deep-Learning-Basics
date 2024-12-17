import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn):
    model.train()  # Переводим модель в режим обучения
    total_loss = 0.0  # Переменная для накопления общей ошибки
    num_batches = 0  # Счетчик количества батчей

    for inputs, targets in data_loader:  # Проходимся по всем батчам в даталоадере
        optimizer.zero_grad()  # Обнуляем градиенты

        outputs = model(inputs)  # Прямой проход (forward pass)
        loss = loss_fn(outputs, targets)  # Вычисляем функцию потерь

        loss.backward()  # Обратное распространение (backward pass)
        optimizer.step()  # Шаг оптимизации

        total_loss += loss.item()  # Сохраняем значение функции потерь
        num_batches += 1  # Увеличиваем счетчик батчей

        print(f"{loss.item():.5f}")  # Печатаем ошибку с точностью до 5 символов после запятой

    mean_loss = total_loss / num_batches  # Средняя ошибка за эпоху
    return mean_loss