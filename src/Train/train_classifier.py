import pickle
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.Model.Attention
import src.Model.GPT
import src.Model.Transformer
from src.config import get_config
from src.Model.ClassificationDataLoader import ClassificationDataset
from src.Model.GPT import GPTModel
from src.Model.Tokenizer import ChemTokenizer
from src.utils import get_project_root, load_latest_checkpoint

# Fix for loading checkpoints saved with different directory structure
sys.modules["GPT"] = src.Model.GPT
sys.modules["Transformer"] = src.Model.Transformer
sys.modules["Attention"] = src.Model.Attention


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def calc_metrics_loader(data_loader, model, device, num_batches=None):
    model.eval()
    all_preds = []
    all_targets = []

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            all_preds.extend(predicted_labels.cpu().numpy())
            all_targets.extend(target_batch.cpu().numpy())
        else:
            break

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = np.mean(all_preds == all_targets)
    f1 = f1_score(all_targets, all_preds, average="weighted")

    return accuracy, f1


def load_data(project_root, batch_size=12, test_size=0.2, random_state=42):
    data_path = project_root / "data" / "filtered_classification_data.pickle"

    with data_path.open("rb") as handle:
        data = pickle.load(handle)

    y = np.array(data["y"])
    # If y is one-hot encoded, convert to indices for stratification
    if y.ndim > 1:
        num_classes = y.shape[1]
        y_indices = np.argmax(y, axis=1)
    else:
        num_classes = int(np.max(y) + 1)
        y_indices = y

    indices = np.arange(len(y))
    train_indices, val_indices = train_test_split(
        indices, test_size=test_size, stratify=y_indices, random_state=random_state
    )

    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    tokenizer_path = project_root / "ChemBPETokenizer" / "tokenizer.json"
    tokenizer = ChemTokenizer(tokenizer_path=str(tokenizer_path))

    train_dataset = ClassificationDataset(str(data_path), train_indices, tokenizer)
    val_dataset = ClassificationDataset(str(data_path), val_indices, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, val_loader, num_classes


def train_classifier_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    save_path=None,
):
    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    examples_seen, global_step = 0, -1
    best_val_f1 = -1.0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        with tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
        ) as tepoch:
            for input_batch, target_batch in tepoch:
                optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()  # Calculate loss gradients
                optimizer.step()  # Update model weights using loss gradients
                examples_seen += input_batch.shape[
                    0
                ]  # New: track examples instead of tokens
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    tepoch.set_postfix(
                        train_loss=f"{train_loss:.3f}", val_loss=f"{val_loss:.3f}"
                    )

        # Calculate metrics after each epoch
        train_accuracy, train_f1 = calc_metrics_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy, val_f1 = calc_metrics_loader(
            val_loader, model, device, num_batches=eval_iter
        )

        tqdm.write(
            f"Epoch {epoch + 1}: "
            f"Train Acc: {train_accuracy * 100:.2f}% F1: {train_f1:.3f} | "
            f"Val Acc: {val_accuracy * 100:.2f}% F1: {val_f1:.3f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if save_path:
                torch.save(model.state_dict(), save_path)
                tqdm.write(f"  -> New best model saved with F1: {best_val_f1:.3f}")

        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        scheduler.step()

    return (
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        train_f1s,
        val_f1s,
        examples_seen,
    )


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    # plt.show()


def load_and_train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    gpt_config = get_config()
    project_root = get_project_root()
    checkpoint_dir = project_root / "checkpoints" / "pretraining"

    # Load Data
    train_loader, val_loader, num_classes = load_data(project_root)

    # Initialize Model
    model = GPTModel(gpt_config)
    model.to(device)

    # Load Checkpoint
    epoch = load_latest_checkpoint(model, checkpoint_dir)
    if epoch is None:
        print("Failed to load model. Exiting.")
        return
    print(f"Model loaded from epoch {epoch}")

    # print(model)
    # Validation loop or similar could go here
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Number of classes: {num_classes}")

    torch.manual_seed(123)

    model.out_head = torch.nn.Linear(
        in_features=gpt_config["emb_dim"], out_features=num_classes
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    # Prepare save directory
    save_dir = project_root / "checkpoints" / "classifier"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "classifier.pt"

    num_epochs = 10
    (
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        train_f1s,
        val_f1s,
        examples_seen,
    ) = train_classifier_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=50,
        eval_iter=5,
        save_path=save_path,
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

    plot_values(
        epochs_tensor, examples_seen_tensor, train_losses, val_losses, label="loss"
    )

    # For accuracy and f1, we have fewer points (one per epoch)
    epochs_tensor_acc = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor_acc = torch.linspace(0, examples_seen, len(train_accs))

    plot_values(
        epochs_tensor_acc,
        examples_seen_tensor_acc,
        train_accs,
        val_accs,
        label="accuracy",
    )
    plot_values(
        epochs_tensor_acc,
        examples_seen_tensor_acc,
        train_f1s,
        val_f1s,
        label="f1_score",
    )


if __name__ == "__main__":
    load_and_train_model()
