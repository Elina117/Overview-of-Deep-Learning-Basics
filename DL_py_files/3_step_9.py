from torch import nn

def create_mlp_model():
    model = nn.Sequential(
        nn.Flatten(),  # Преобразуем изображение 28x28 в вектор длины 784
        nn.Linear(28 * 28, 512),  # Первый скрытый слой
        nn.ReLU(),  # Функция активации ReLU
        nn.Linear(512, 256),  # Второй скрытый слой
        nn.ReLU(),  # Функция активации ReLU
        nn.Linear(256, 10)  # Выходной слой для 10 классов
    )
    return model