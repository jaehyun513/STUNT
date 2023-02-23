import torch.nn as nn

class MLPProto(nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes, drop_p = 0.):
        super(MLPProto, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes
        self.drop_p = drop_p

        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_sizes, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_sizes, hidden_sizes, bias=True)
        )

    def forward(self, inputs):
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[2:]))
        return embeddings.view(*inputs.shape[:2], -1)