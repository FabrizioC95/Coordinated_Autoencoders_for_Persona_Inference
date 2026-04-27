import torch
import pandas as pd


def run_inference(model, dataloader, dataset, device, k):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch, indices in dataloader:
            batch = batch.to(device)

            _, _, probs = model(batch)

            best_autoencoder = torch.argmax(probs, dim=1).cpu().numpy()
            indices = indices.cpu().numpy()

            for idx, cluster in zip(indices, best_autoencoder):
                predictions.append((idx, cluster))

    results = pd.DataFrame(predictions, columns=['Index', 'Cluster'])
    return results
