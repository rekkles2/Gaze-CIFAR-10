
import torch
import torch.nn as nn

class DSGE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, target_sequence_length, num_heads, num_layers):
        super(DSGE, self).__init__()
        self.hidden_size = hidden_size
        self.target_sequence_length = target_sequence_length


        self.input_fc = nn.Linear(input_size, hidden_size)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
            ),
            num_layers=num_layers
        )


        self.conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)


        self.fc = nn.Linear(hidden_size, output_size)

        self.pre_logits = nn.Identity()

    def forward(self, x):

        x = self.input_fc(x)

        x = x.permute(1, 0, 2)

        x = self.transformer(x)

        if x.size(0) != self.target_sequence_length:
            x = x.permute(1, 2, 0)
            x = x.permute(2, 0, 1)

        x = self.fc(x.permute(1, 0, 2))

        x = self.pre_logits(x[:, 0])

        return x


