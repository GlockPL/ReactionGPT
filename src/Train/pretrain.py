# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import os

# Disable tokenizers parallelism to avoid deadlocks/segfaults in DDP/multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from GPTDataloader import create_dataloader_v1
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from config import get_config
from src.Model.GPT import GPTModel, generate_text_simple
from src.Model.Tokenizer import ChemTokenizer


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded.ids).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    # Handle DDP wrapped model
    if isinstance(model, DDP):
        model_ref = model.module
    else:
        model_ref = model

    context_size = model_ref.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model_ref, idx=encoded, max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        # Only print on rank 0
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    checkpoint_dir="checkpoints",
):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Check for DDP
    is_ddp = dist.is_initialized()
    rank = dist.get_rank() if is_ddp else 0

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        # Set epoch for DistributedSampler
        if is_ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # Only show tqdm on rank 0
        if rank == 0:
            batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        else:
            batch_iterator = train_loader

        postfix_values = {}

        for input_batch, target_batch in batch_iterator:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            if rank == 0:
                # Update progress bar with loss
                postfix_values["loss"] = f"{loss.item():.4f}"
                if isinstance(batch_iterator, tqdm):
                    batch_iterator.set_postfix(postfix_values)

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )

                if rank == 0:
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)

                    # Update validation metrics in the progress bar
                    postfix_values["val_loss"] = f"{val_loss:.3f}"
                    postfix_values["train_eval"] = f"{train_loss:.3f}"
                    if isinstance(batch_iterator, tqdm):
                        batch_iterator.set_postfix(postfix_values)

        # Print a sample text after each epoch (handled inside to respect rank)
        generate_and_print_sample(model, tokenizer, device, start_context)

        # Save Checkpoint
        if rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            raw_model = model.module if is_ddp else model
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "config": raw_model.parameters,  # Storing config might be tricky if not stored in model, but we have it elsewhere.
            }
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
            # Also save as latest
            torch.save(checkpoint, os.path.join(checkpoint_dir, "checkpoint_latest.pt"))
            if isinstance(batch_iterator, tqdm):
                batch_iterator.write(f"Checkpoint saved to {checkpoint_path}")

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


def main(gpt_config, settings):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################
    # Load data
    ##############################

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "pretraining_data.pickle"

    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        return

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    reactions = data["reactions"]
    groups = data["groups"]

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    model.to(
        device
    )  # no assignment model = model.to(device) necessary for nn.Module classes
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(reactions))

    train_loader = create_dataloader_v1(
        reactions[:split_idx],
        groups[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader_v1(
        reactions[split_idx:],
        groups[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    ##############################
    # Train model
    ##############################

    tokenizer_path = project_root / "ChemBPETokenizer" / "tokenizer.json"
    tokenizer = ChemTokenizer(tokenizer_path=str(tokenizer_path))

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=settings["num_epochs"],
        eval_freq=5,
        eval_iter=1,
        start_context='Reaction:"ClS(C1=CC=CC=C1)(=O)=O.C=CC[B-](F)(F)F>>O=S(C2=CC=CC=C2)(CC=C)=O" Group:"',
        tokenizer=tokenizer,
    )

    return train_losses, val_losses, tokens_seen, model


# -----------------------------------------------------------------------------
# DDP Setup & Main
# -----------------------------------------------------------------------------


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # nccl is recommended for NVIDIA GPUs
    # Using gloo to avoid SIGSEGV on some environments
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main_ddp(rank, world_size, gpt_config, settings):
    print(f"[Rank {rank}] Setting up DDP...")
    ddp_setup(rank, world_size)
    print(f"[Rank {rank}] DDP Setup complete.")

    # 1. Prepare data (Loaded by all ranks for now, can be optimized)
    # Robustly find data path
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "pretraining_data.pickle"

    if not data_path.exists():
        if rank == 0:
            print(f"Data file not found at {data_path}")
        cleanup()
        return

    print(f"[Rank {rank}] Loading data...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    print(f"[Rank {rank}] Data loaded.")

    reactions = data["reactions"]
    groups = data["groups"]

    # 2. Initialize Model
    print(f"[Rank {rank}] Initializing model structure...")
    model = GPTModel(gpt_config)
    print(f"[Rank {rank}] Model structure initialized. Moving to device {rank}...")
    model.to(rank)  # move to specific GPU
    print(f"[Rank {rank}] Model on device. Wrapping in DDP...")
    model = DDP(model, device_ids=[rank])
    print(f"[Rank {rank}] Model initialized and wrapped.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )

    # 3. Dataloaders with DistributedSampler
    train_ratio = 0.90
    split_idx = int(train_ratio * len(reactions))

    train_dataset_reactions = reactions[:split_idx]
    train_dataset_groups = groups[:split_idx]

    # Actually, DistributedSampler needs the dataset. But create_dataloader_v1 instantiates it.
    # We can't easily pass the sampler if we don't have the dataset yet.
    # Let's instantiate the dataset first here.

    tokenizer_path = project_root / "ChemBPETokenizer" / "tokenizer.json"
    tokenizer = ChemTokenizer(tokenizer_path=str(tokenizer_path))

    # NOTE: Loading dataset on every rank is inefficient but simple for now.
    from GPTDataloader import GPTDatasetV1

    print(f"[Rank {rank}] Creating Training Dataset (Tokenizing)...")
    train_ds = GPTDatasetV1(
        train_dataset_reactions,
        train_dataset_groups,
        tokenizer,
        gpt_config["context_length"],
        gpt_config["context_length"],
    )
    print(f"[Rank {rank}] Training Dataset created.")

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=settings["batch_size"],
        shuffle=False,  # Sampler handles shuffling
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )

    val_dataset_reactions = reactions[split_idx:]
    val_dataset_groups = groups[split_idx:]

    val_ds = GPTDatasetV1(
        val_dataset_reactions,
        val_dataset_groups,
        tokenizer,
        gpt_config["context_length"],
        gpt_config["context_length"],
    )

    # Validation sampler is optional but good for distributed eval
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=settings["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True,
    )

    # 4. Train
    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        rank,  # device is rank int for cuda:rank
        num_epochs=settings["num_epochs"],
        eval_freq=5,
        eval_iter=1,
        start_context='Reaction:"ClS(C1=CC=CC=C1)(=O)=O.C=CC[B-](F)(F)F>>O=S(C2=CC=CC=C2)(CC=C)=O" Group:"',
        tokenizer=tokenizer,
    )

    if rank == 0:
        # Plot results
        epochs_tensor = torch.linspace(0, settings["num_epochs"], len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        plt.savefig("loss.pdf")

        # Save model
        torch.save(model.module.state_dict(), "model.pth")
        print("Model saved to model.pth")

    cleanup()


def run_ddp(world_size=None):
    if world_size is None:
        world_size = torch.cuda.device_count()

    model_config = get_config()
    train_config = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 4,  # Per GPU
        "weight_decay": 0.1,
    }

    print(f"Spawning {world_size} processes for DDP training...")
    mp.spawn(main_ddp, args=(world_size, model_config, train_config), nprocs=world_size)


def full_run():
    model_config = get_config()

    train_config = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 5,
        "weight_decay": 0.1,
    }

    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, model = main(model_config, train_config)

    ###########################
    # After training
    ###########################

    # Plot results
    epochs_tensor = torch.linspace(0, train_config["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # Save and load model
    torch.save(model.state_dict(), "model.pth")
    model = GPTModel(model_config)
    model.load_state_dict(torch.load("model.pth", weights_only=True))


if __name__ == "__main__":
    # Simple check to allow user to run DDP via flag or comment

    if "--ddp" in sys.argv:
        run_ddp()
    else:
        full_run()
