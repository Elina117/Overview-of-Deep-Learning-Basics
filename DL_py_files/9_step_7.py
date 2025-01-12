import torch
import torch.nn as nn


class Similarity1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_states: torch.Tensor, decoder_state: torch.Tensor):
        # encoder_states.shape = [T, N] (T - временные шаги, N - размерность векторов)
        # decoder_state.shape = [N] (N - размерность вектора декодера)

        similarity = torch.matmul(encoder_states, decoder_state)
        return similarity
