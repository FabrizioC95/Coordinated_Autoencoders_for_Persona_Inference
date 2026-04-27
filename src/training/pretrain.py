import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans

from src.model.clustering_head import MixtureAssignmentNetwork


class MixtureDataLoader(Dataset):
    def __init__(self, dataframe):
        self.features = dataframe.drop(columns=['pseudo_labels']).values
        self.pseudo_labels = dataframe['pseudo_labels'].values

        self.X = torch.tensor(self.features, dtype=torch.float32)
        self.y = torch.tensor(self.pseudo_labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], idx, self.y[idx]


def shallow_pt_first(k, input_features, targets=None, model='kmeans', generator=None, random_seed=None):
    if isinstance(input_features, pd.DataFrame):
        input_features = input_features.to_numpy()

    if isinstance(input_features, torch.Tensor):
        input_features = input_features.numpy()

    if generator is not None and random_seed is not None:
        raise ValueError("Both 'generator' and 'random_seed' arguments cannot be provided at the same time")

    if generator is not None:
        random_state = generator.initial_seed()
    elif random_seed is not None:
        random_state = random_seed
    else:
        random_state = random_seed

    if model == 'kmeans':
        p_kmeans = KMeans(n_clusters=k, random_state=random_state)
        pseudo_labels = p_kmeans.fit_predict(input_features)

    feature_columns = [f"feature_{i}" for i in range(input_features.shape[1])]
    aligned_df = pd.DataFrame(input_features, columns=feature_columns)
    aligned_df['pseudo_labels'] = pseudo_labels

    return aligned_df


def pretrain_mixture_assignment_network(k, pseudo_data, data_dim, cluster_hidden_sizes,
                                        batch_normalize=True, pt_num_epochs=10,
                                        pt_batch_size=64, pre_lr=0.001, weight_decay=0.001,
                                        generator=None, device=None):
    mixture_assignment_net = MixtureAssignmentNetwork(
        k=k,
        data_dim=data_dim,
        cluster_hidden_sizes=cluster_hidden_sizes,
        batch_normalize=batch_normalize
    ).to(device)

    dataset = MixtureDataLoader(pseudo_data)
    dataloader = DataLoader(dataset, batch_size=pt_batch_size, shuffle=True, generator=generator)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(mixture_assignment_net.parameters(), lr=pre_lr, weight_decay=weight_decay)

    mixture_assignment_net.train()

    for epoch in range(pt_num_epochs):
        for batch_data, _, pseudo_labels in dataloader:
            optimizer.zero_grad()
            batch_data, pseudo_labels = batch_data.to(device), pseudo_labels.to(device)

            outputs = mixture_assignment_net(batch_data)
            loss = criterion(outputs, pseudo_labels)
            loss.backward()
            optimizer.step()

    return mixture_assignment_net
