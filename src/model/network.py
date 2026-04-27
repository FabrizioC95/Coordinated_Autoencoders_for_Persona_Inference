import torch.nn as nn
from src.model.autoencoder import KAutoEncoders
from src.model.clustering_head import MixtureAssignmentNetwork


class ClusteringAutoEncoder(nn.Module):
    def __init__(self, k, data_dim, hidden_dim, cluster_hidden_sizes,
                 batch_normalize=False, cluster_batch_normalize=False, dropout=[False, 0.0]):
        super(ClusteringAutoEncoder, self).__init__()

        self.k_autoencoders = KAutoEncoders(
            k=k,
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            batch_normalize=batch_normalize,
            dropout=dropout
        )

        self.cluster_net = MixtureAssignmentNetwork(
            k=k,
            data_dim=data_dim,
            cluster_hidden_sizes=cluster_hidden_sizes,
            batch_normalize=cluster_batch_normalize
        )

    def forward(self, x):
        embeddings, reconstructions = self.k_autoencoders(x)
        cluster_probs = self.cluster_net(x)

        return embeddings, reconstructions, cluster_probs
        # embeddings:      (batch_size, k, embedding_dim)
        # reconstructions: (batch_size, k, data_dim)
        # cluster_probs:   (batch_size, k)
