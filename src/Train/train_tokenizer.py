import pickle
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def train_tokenizer():
    data_path = "../../data/pretraining_data.pickle"
    print(f"Loading data from {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    reactions = data["reactions"]
    groups = data["groups"]

    if len(reactions) != len(groups):
        print(
            f"Warning: Reactions length ({len(reactions)}) does not match Groups length ({len(groups)})"
        )
        return

    print(f"Training on {len(reactions)} examples.")

    def data_iterator():
        for i, (r, g) in enumerate(zip(reactions, groups)):
            yield f'Reaction:"{r}" Group:"{g}"'

    # Initialize a tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Use Whitespace pre-tokenizer to handle the spaces in the format
    tokenizer.pre_tokenizer = Whitespace()

    # Initialize the trainer
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<|endoftext|>"],
        vocab_size=30000,
        show_progress=True,
    )

    print("Starting training...")
    tokenizer.train_from_iterator(data_iterator(), trainer=trainer)

    save_path = Path("../../ChemBPETokenizer/tokenizer.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))
    print(f"Tokenizer saved to {save_path}")


if __name__ == "__main__":
    train_tokenizer()
