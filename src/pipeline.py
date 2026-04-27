import torch
import torch.optim as optim

from src.data.dataloader import load_data
from src.model.network import ClusteringAutoEncoder
from src.training.pretrain import shallow_pt_first, pretrain_mixture_assignment_network
from src.training.trainer import samplewise_trainer
from src.utils.seed import reset_seed
from src.utils.inference import run_inference


def train_model(data,
                k,
                categorical_cols,
                numerical_cols,
                batch_size,
                hidden_dim=[128, 64, 32],
                cluster_hidden_sizes=[64, 32],
                num_epochs=100,
                pt_num_epochs=10,
                lr=0.001,
                seed=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: Using {device} for training")

    generator = reset_seed(seed)

    print("Initializing dataloader..")
    dataset, dataloader, k, data_dim, df = load_data(
        df=data,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        k=k,
        batch_size=batch_size,
        generator=generator
    )

    pseudo_data = shallow_pt_first(k=k, input_features=df, model='kmeans', generator=generator)

    pretrained_man = pretrain_mixture_assignment_network(
        k=k,
        pseudo_data=pseudo_data,
        data_dim=data_dim,
        cluster_hidden_sizes=cluster_hidden_sizes,
        batch_normalize=True,
        pt_num_epochs=pt_num_epochs,
        pt_batch_size=batch_size,
        pre_lr=0.001,
        weight_decay=0.001,
        generator=generator,
        device=device
    )

    model = ClusteringAutoEncoder(
        k=k,
        data_dim=data_dim,
        hidden_dim=hidden_dim,
        cluster_hidden_sizes=cluster_hidden_sizes,
        batch_normalize=True,
        cluster_batch_normalize=True,
        dropout=[False, 0.0]
    ).to(device)

    model.cluster_net.load_state_dict(pretrained_man.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Training Network..")
    trained_model = samplewise_trainer(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        optimizer=optimizer,
        num_epochs=num_epochs,
        alpha=5,
        beta=5,
        k=k,
        device=device,
        schedule="batch"
    )

    print("Running inference..")
    inference_df = run_inference(
        model=trained_model,
        dataloader=dataloader,
        dataset=dataset,
        device=device,
        k=k
    )

    results_df = df.copy()
    results_df['Cluster'] = results_df.index.map(inference_df.set_index('Index')['Cluster'])

    return results_df
