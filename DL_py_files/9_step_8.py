import torch
import torch.nn as nn


class Similarity2(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, intermediate_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(encoder_dim, intermediate_dim)
        self.fc2 = nn.Linear(decoder_dim, intermediate_dim)
        self.fc3 = nn.Linear(intermediate_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, encoder_states: torch.Tensor, decoder_state: torch.Tensor):
        # encoder_states.shape = [T, N]
        # decoder_state.shape = [N]

        h = self.fc1(encoder_states)  # [T, intermediate_dim]
        s = self.fc2(decoder_state)  # [intermediate_dim]

        f_act = torch.tanh(h + s)

        sim = self.fc3(f_act)

        return sim