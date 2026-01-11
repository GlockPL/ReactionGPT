# ChemGPT

ChemGPT is a transformer-based Large Language Model (LLM) designed to understand and generate chemical reaction data. It uses a GPT architecture trained from scratch on SMILES representations of chemical reactions and functional groups.

## Features

*   **Custom Tokenization**: Uses a Byte-Pair Encoding (BPE) tokenizer specifically trained on chemical SMILES strings.
*   **GPT Architecture**: Implements a standard decoder-only transformer with Multi-Head Flash Attention.
*   **Distributed Training**: Supports Distributed Data Parallel (DDP) for efficient training across multiple GPUs.
*   **Checkpointing**: Automatically saves model states, optimizer states, and loss history after every epoch.
*   **Live Monitoring**: Real-time loss tracking and validation metrics via `tqdm`.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/ChemGPT.git
    cd ChemGPT
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    ```
    *Alternatively, if not using uv:*
    ```bash
    pip install torch tokenizers tqdm matplotlib
    ```

## Data Preparation

The model expects a pickle file located at `data/pretraining_data.pickle`. This file should contain a dictionary with two lists:
*   `reactions`: List of SMILES strings representing reactions.
*   `groups`: List of SMILES strings representing associated groups.

**Input Format**:
The model processes data in the following format:
```text
Reaction:"<Reaction_SMILES>" Group:"<Group_SMILES>"
```

## Usage

### 1. Train Tokenizer (Optional)
If you need to retrain the tokenizer on new data:
```bash
uv run python src/train_tokenizer.py
```
This saves the tokenizer to `ChemBPETokenizer/tokenizer.json`.

### 2. Pre-train the Model

**Single GPU / CPU:**
```bash
uv run python src/pretrain.py
```

**Multi-GPU (DDP):**
To utilize multiple GPUs via Distributed Data Parallel:
```bash
uv run python src/pretrain.py --ddp
```

### 3. Checkpoints
Checkpoints are saved automatically to the `checkpoints/` directory:
*   `checkpoint_epoch_N.pt`: State at end of epoch N.
*   `checkpoint_latest.pt`: Always contains the most recent state.

## Testing

Run the test suite to ensure the context window and tokenizer are configured correctly:

```bash
uv run pytest tests/
```

## Project Structure

```text
ChemGPT/
├── ChemBPETokenizer/   # Trained tokenizer artifacts
├── checkpoints/        # Saved model checkpoints
├── data/               # Input data (pickle)
├── src/
│   ├── Attention.py    # Multi-Head Attention implementation
│   ├── GPT.py          # Main model architecture
│   ├── GPTDataloader.py# Dataset and Dataloader logic
│   ├── Tokenizer.py    # Tokenizer wrapper
│   ├── Transformer.py  # Transformer block implementation
│   ├── config.py       # Model configuration
│   └── pretrain.py     # Main training script (Single & DDP)
├── tests/              # Unit tests
├── pyproject.toml      # Dependency configuration
└── README.md
```

## Credits

Based on the architecture and training loops from *"Build a Large Language Model From Scratch"* by Sebastian Raschka.
