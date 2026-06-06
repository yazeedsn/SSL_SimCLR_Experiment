from tqdm import tqdm 
from datasets import get_labeled_ds
from torch.utils.data import DataLoader
import torch 

def train_linear_prob(model, optimizer, loss_fn, device, n_epochs):
    ds = get_labeled_ds()
    train_dl = DataLoader(ds, batch_size=256)

    loss_hist = []
    acc_hist = []
    model.train()
    for epoch in range(n_epochs):
        total = 0
        running_loss = 0.0
        true_predictions = 0
        bbar = tqdm(enumerate(train_dl))
        for batch_idx, (X, Y) in bbar:
            X, Y = X.to(device), Y.to(device)
            Yp = model(X)
            loss = loss_fn(Yp, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += len(X)
            true_predictions += (Y == Yp.argmax(dim=1)).to(torch.uint8).sum()
            running_loss += loss.item()
            bbar.set_description(f"Epoch [{epoch+1}/{n_epochs}] Batch [{batch_idx+1}/{len(train_dl)}]")
            bbar.set_postfix(Loss=f"{loss.item():.4f}", step=batch_idx+1)

        avg_loss = running_loss / len(train_dl)
        loss_hist.append(avg_loss)
        acc_hist.append((true_predictions/total).detach().cpu())
        print(
            f"\nEpoch {epoch+1} completed "
            f"- Average Loss: {avg_loss:.4f}\n"
        )