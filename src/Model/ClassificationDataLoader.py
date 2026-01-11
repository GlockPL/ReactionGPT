import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, pickle_file_path, indices, tokenizer, max_length=None):
        with Path(pickle_file_path).open("rb") as handle:
            data = pickle.load(handle)

        y_data = np.array(data["y"])
        # If y is one-hot encoded, convert to indices
        if y_data.ndim > 1:
            y_data = np.argmax(y_data, axis=1)
            
        self.y = y_data[indices]
        reacts = np.array(data["reactions"])[indices]
        groups = np.array(data["groups"])[indices]
        batch_size = 50000
        current_batch_texts = []
        self.encoded_texts = []

        # Collect all texts
        for r, g in zip(reacts, groups):
            text = f'Reaction:"{r}" Group:"{g}"'
            current_batch_texts.append(text)

            if len(current_batch_texts) >= batch_size:
                # Use the underlying tokenizer's batch encoding
                encodings = tokenizer.tokenizer.encode_batch(current_batch_texts)
                for enc in encodings:
                    self.encoded_texts.append(enc.ids)
                current_batch_texts = []
                print(f"Tokenized {len(self.encoded_texts)} samples...", end="\r")

        # Process remaining texts
        if current_batch_texts:
            encodings = tokenizer.tokenizer.encode_batch(current_batch_texts)
            for enc in encodings:
                self.encoded_texts.append(enc.ids)
            print(f"Tokenized {len(self.encoded_texts)} samples...")

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

        # Truncate sequences if they are longer than max_length
        # (Though usually we want to keep them or handle truncation in tokenizer,
        # doing it here as per previous logic pattern)
        self.encoded_texts = [
            encoded_text[: self.max_length] for encoded_text in self.encoded_texts
        ]

        # Pad sequences to the longest sequence
        pad_token_id = 3
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.y[index]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.y)

    def _longest_encoded_length(self):
        return max(len(encoded_text) for encoded_text in self.encoded_texts)
