import torch.nn as nn


class MixtureAssignmentNetwork(nn.Module):
    def __init__(self, k, data_dim, cluster_hidden_sizes, batch_normalize=False):
        super(MixtureAssignmentNetwork, self).__init__()

        layers = []
        in_dim = data_dim

        for hidden_size in cluster_hidden_sizes:
            linear_layer = nn.Linear(in_dim, hidden_size)
            nn.init.xavier_uniform_(linear_layer.weight)
            layers.append(linear_layer)

            if batch_normalize:
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(nn.ELU())
            in_dim = hidden_size

        output_layer = nn.Linear(in_dim, k)
        nn.init.xavier_uniform_(output_layer.weight)
        layers.append(output_layer)
        layers.append(nn.Softmax(dim=1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
