import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, data_dim, hidden_dim, batch_normalize=False, dropout=[False, 0.2]):
        super(AutoEncoder, self).__init__()

        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.batch_normalize = batch_normalize
        self.dropout = dropout[0]
        self.dropout_p = dropout[1]

        encoder_layers = []
        input_size = data_dim

        for h_dim in hidden_dim[:-1]:
            layer = nn.Linear(input_size, h_dim)
            nn.init.xavier_uniform_(layer.weight)
            encoder_layers.append(layer)

            if batch_normalize:
                encoder_layers.append(nn.BatchNorm1d(h_dim))

            encoder_layers.append(nn.ELU())

            if self.dropout:
                encoder_layers.append(nn.Dropout(p=self.dropout_p))

            input_size = h_dim

        self.embedding_layer = nn.Linear(hidden_dim[-2], hidden_dim[-1])
        nn.init.xavier_uniform_(self.embedding_layer.weight)

        decoder_layers = []
        input_size = hidden_dim[-1]

        for h_dim in reversed(hidden_dim[:-1]):
            layer = nn.Linear(input_size, h_dim)
            nn.init.xavier_uniform_(layer.weight)
            decoder_layers.append(layer)

            if batch_normalize:
                decoder_layers.append(nn.BatchNorm1d(h_dim))

            decoder_layers.append(nn.ELU())

            if self.dropout:
                decoder_layers.append(nn.Dropout(p=self.dropout_p))

            input_size = h_dim

        self.final_layer = nn.Sequential(
            nn.Linear(input_size, data_dim),
            nn.Sigmoid()
        )
        nn.init.xavier_uniform_(self.final_layer[0].weight)

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        embedding = self.embedding_layer(encoded)
        decoded = self.decoder(embedding)
        output = self.final_layer(decoded)
        return embedding, output


class KAutoEncoders(nn.Module):
    def __init__(self, k, data_dim, hidden_dim, batch_normalize=False, dropout=[False, 0.0]):
        super(KAutoEncoders, self).__init__()
        self.k = k
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.autoencoders = nn.ModuleList(
            [AutoEncoder(data_dim, hidden_dim, batch_normalize, dropout=self.dropout)
             for _ in range(k)]
        )

    def forward(self, x):
        embeddings = []
        reconstructions = []

        for autoencoder in self.autoencoders:
            embedding, reconstruction = autoencoder(x)
            embeddings.append(embedding)
            reconstructions.append(reconstruction)

        reconstructions = torch.stack(reconstructions, dim=1)  #(batc_size, k, data_dim)
        embeddings = torch.stack(embeddings, dim=1)            #(batch_size, k, embedding_dim)

        return embeddings, reconstructions
