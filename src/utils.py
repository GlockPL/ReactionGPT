from pathlib import Path

import torch


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def load_latest_checkpoint(model, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return None

    # Try to load 'checkpoint_latest.pt' first
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    if latest_path.exists():
        print(f"Loading latest checkpoint: {latest_path}")
        checkpoint = torch.load(
            latest_path, map_location=torch.device("cpu"), weights_only=False
        )
    else:
        # Find the checkpoint with the highest epoch number
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            print("No checkpoints found.")
            return None

        # Sort by epoch number assuming format 'checkpoint_epoch_N.pt'
        def extract_epoch(p):
            try:
                return int(p.stem.split("_")[-1])
            except ValueError:
                return -1

        latest_path = max(checkpoints, key=extract_epoch)
        print(f"Loading checkpoint: {latest_path}")
        checkpoint = torch.load(
            latest_path, map_location=torch.device("cpu"), weights_only=False
        )

    # Load state dict
    # Handle DDP prefix 'module.' if present
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return checkpoint.get("epoch", "unknown")
