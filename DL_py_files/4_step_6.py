from torchvision import transforms as T


def get_augmentations(train: bool = True) -> T.Compose:
    means = (0.49139968, 0.48215841, 0.44653091)
    stds = (0.24703223, 0.24348513, 0.26158784)

    if train:
        return T.Compose([
            T.Resize((224, 224)),  # Изменение размера изображения
            T.RandomAdjustSharpness(sharpness_factor=2),  # Изменение резкости
            T.RandomHorizontalFlip(),  # Случайное горизонтальное отражение
            T.RandomRotation(15),  # Случайный поворот на ±15 градусов
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Изменение цвета
            T.ToTensor(),  # Преобразование в тензор
            T.Normalize(means, stds)  # Нормализация
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),  # Изменение размера изображения
            T.ToTensor(),  # Преобразование в тензор
            T.Normalize(means, stds)  # Нормализация
        ])
