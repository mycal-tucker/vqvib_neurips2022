import torch.nn as nn
import torch


class Team(nn.Module):
    def __init__(self, speaker, listener, decoder):
        super(Team, self).__init__()
        self.speaker = speaker
        self.listener = listener
        self.decoder = decoder

    def forward(self, speaker_x, listener_x):
        comm, speaker_loss, info = self.speaker(speaker_x)
        full_listener_input = torch.hstack([comm, listener_x])
        prediction, _, _ = self.listener(full_listener_input)
        recons = self.decoder(comm)
        return prediction, speaker_loss, info, recons
