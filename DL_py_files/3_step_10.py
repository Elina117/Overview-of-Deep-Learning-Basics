from torch import nn


def create_conv_model():
    model = nn.Sequential(
        # Первый сверточный слой
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # Padding чтобы сохранить размер
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Второй сверточный слой
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # Padding чтобы сохранить размер
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Третий сверточный слой
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # Padding чтобы сохранить размер
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Четвертый сверточный слой
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # Padding чтобы сохранить размер
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Полносвязные слои
        nn.Flatten(),
        nn.Linear(256 * 1 * 1, 512),  # Уменьшили размер после последнего слоя свертки
        nn.ReLU(),
        nn.Linear(512, 10)  # 10 классов для MNIST
    )
    return model
