import torch


def function02(dataset: torch.Tensor) -> torch.Tensor:
    num_features = dataset.shape[1]
    weights = torch.rand(num_features, dtype=torch.float32, requires_grad=True)
    return weights


def function03(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    weights = function02(x)

    step = 1e-2
    max_epochs = 1000
    target_loss = 1.0

    for epoch in range(max_epochs):
        y_pred = x @ weights

        loss = torch.mean((y_pred - y) ** 2)

        if loss.item() < target_loss:
            break

        # Вычисляем градиенты
        loss.backward()

        # Обновляем веса
        with torch.no_grad():  # Отключаем подсчет градиентов
            weights -= step * weights.grad  # Обновляем веса
            weights.grad.zero_()  # Обнуляем градиенты для следующей итерации

    return weights.detach()  # Возвращаем веса без градиентов
