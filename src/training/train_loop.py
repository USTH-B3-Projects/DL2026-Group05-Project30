import os
import torch
from src.training.config import DEVICE, CHECKPOINT_DIR

def train_one_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, all_preds, all_labels

def fit(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    num_epochs,
    model_name: str = None,
    patience: int = None,
):
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, _, _ = evaluate(model, val_loader, loss_fn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val_loss
        print(
            f"Epoch {epoch}/{num_epochs} - train_loss: {train_loss:.4f} - "
            f"val_loss: {val_loss:.4f}" + ("  (best so far)" if improved else "")
        )

        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if model_name:
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")
                torch.save(model.state_dict(), path)
        else:
            epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                print(
                    f"No improvement for {patience} epochs -- stopping early "
                    f"at epoch {epoch}/{num_epochs}."
                )
                break

    return history
