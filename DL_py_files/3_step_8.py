def count_parameters_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool):
    # Подсчет параметров для весов и сдвига по формуле: (C1 * n^2 + 1) * C2
    num_parameters = (in_channels * kernel_size ** 2 + (1 if bias else 0)) * out_channels
    return num_parameters