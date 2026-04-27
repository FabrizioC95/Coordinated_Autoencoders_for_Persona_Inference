import torch


def samplewise_trainer(model, dataloader, dataset, optimizer, num_epochs, alpha, beta, k, device, schedule='batch'):
    model.train()

    for epoch in range(num_epochs):
        for batch, indices in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            embeddings, reconstructions, probs = model(batch)

            batch_expanded = batch.unsqueeze(1)
            diff = batch_expanded - reconstructions
            l2_norm = torch.sum(diff ** 2, dim=-1)
            weighted_errors = probs * l2_norm
            ae_losses = torch.sum(weighted_errors, dim=1)

            sample_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

            avg_probs = probs.mean(dim=0)
            batch_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))

            total_loss = (torch.sum(ae_losses + alpha * sample_entropy) / batch.size(0)) - beta * batch_entropy

            total_loss.backward()
            optimizer.step()

            if schedule == 'batch':
                with torch.no_grad():
                    alpha = 1
                    samplewise_term = torch.sum(ae_losses + alpha * sample_entropy) / batch.size(0)
                    batch_entr_magnitude = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
                    beta = samplewise_term / (batch_entr_magnitude + 1e-8)

        if schedule == 'epoch':
            with torch.no_grad():
                alpha = 1
                samplewise_term = torch.sum(ae_losses + alpha * sample_entropy) / batch.size(0)
                batch_entr_magnitude = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
                beta = samplewise_term / (batch_entr_magnitude + 1e-8)

    return model
