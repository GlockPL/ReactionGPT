import torch
from torch.utils.data import DataLoader, Dataset

from src.Model.Tokenizer import ChemTokenizer
from src.utils import get_project_root


class GPTDatasetV1(Dataset):
    def __init__(self, reactions, groups, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Get the ID for the <|endoftext|> token (if it exists)
        # We assume it was added during training.
        eos_token_id = tokenizer.tokenizer.token_to_id("<|endoftext|>")
        if eos_token_id is None:
            # Fallback if not found (though user added it previously)
            print(
                "Warning: <|endoftext|> token not found in tokenizer. Using [SEP] or 0."
            )
            eos_token_id = tokenizer.tokenizer.token_to_id("[SEP]") or 0

        # Construct the full sequence of token IDs
        full_token_ids = []

        # We process in batches to be efficient with memory and speed
        batch_size = 50000
        current_batch_texts = []

        print(f"Tokenizing {len(reactions)} samples...")

        for r, g in zip(reactions, groups):
            text = f'Reaction:"{r}" Group:"{g}"'
            current_batch_texts.append(text)

            if len(current_batch_texts) >= batch_size:
                # Use the underlying tokenizer's batch encoding
                encodings = tokenizer.tokenizer.encode_batch(current_batch_texts)
                for enc in encodings:
                    full_token_ids.extend(enc.ids)
                    full_token_ids.append(eos_token_id)
                current_batch_texts = []
                print(f"Tokenized {len(full_token_ids)} tokens...", end="\r")

        # Process remaining
        if current_batch_texts:
            encodings = tokenizer.tokenizer.encode_batch(current_batch_texts)
            for enc in encodings:
                full_token_ids.extend(enc.ids)
                full_token_ids.append(eos_token_id)

        print(f"\nTotal tokens: {len(full_token_ids)}")

        # Use a sliding window to chunk into overlapping sequences
        # Note: This loads the entire dataset into RAM as tensors.
        # For 3M reactions -> ~150M tokens -> ~600MB RAM for the list.
        # The tensors will take more.

        for i in range(0, len(full_token_ids) - max_length, stride):
            input_chunk = full_token_ids[i : i + max_length]
            target_chunk = full_token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    reactions,
    groups,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    sampler=None,
):
    project_root = get_project_root()
    tokenizer_path = project_root / "ChemBPETokenizer" / "tokenizer.json"

    tokenizer = ChemTokenizer(tokenizer_path=str(tokenizer_path))

    # Create dataset
    dataset = GPTDatasetV1(reactions, groups, tokenizer, max_length, stride)

    # If sampler is provided, shuffle must be False for DataLoader
    if sampler is not None:
        shuffle = False

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        sampler=sampler,
    )

    return dataloader
