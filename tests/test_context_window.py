import pickle
import sys
from pathlib import Path

import pytest

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.Model.Tokenizer import ChemTokenizer

# Constants
CONTEXT_WINDOW = 2048
DATA_PATH = project_root / "data" / "pretraining_data.pickle"
TOKENIZER_PATH = project_root / "ChemBPETokenizer" / "tokenizer.json"


@pytest.fixture(scope="module")
def loaded_data():
    if not DATA_PATH.exists():
        pytest.fail(f"Data file not found at {DATA_PATH}")
    print(f"\nLoading data from {DATA_PATH}...")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    return data


@pytest.fixture(scope="module")
def tokenizer():
    if not TOKENIZER_PATH.exists():
        pytest.fail(f"Tokenizer file not found at {TOKENIZER_PATH}")
    return ChemTokenizer(str(TOKENIZER_PATH))


def test_longest_reaction_by_chars(loaded_data, tokenizer):
    """
    Finds the longest reaction by character count and verifies its token length.
    """
    reactions = loaded_data["reactions"]
    groups = loaded_data["groups"]

    longest_text = ""
    max_char_len = 0

    print("\nScanning for longest character sequence...")
    for r, g in zip(reactions, groups):
        text = f'Reaction:"{r}" Group:"{g}"'
        if len(text) > max_char_len:
            max_char_len = len(text)
            longest_text = text

    encoded = tokenizer.encode(longest_text)
    num_tokens = len(encoded.ids)

    print(f"Longest text (chars): {max_char_len}")
    print(f"Token count: {num_tokens}")

    assert num_tokens < CONTEXT_WINDOW, (
        f"Longest char reaction ({num_tokens} tokens) exceeds limit {CONTEXT_WINDOW}"
    )


def test_all_reactions_within_limit(loaded_data, tokenizer):
    """
    Checks EVERY reaction to ensure it fits within the context window.
    Uses batch processing for efficiency.
    """
    reactions = loaded_data["reactions"]
    groups = loaded_data["groups"]

    batch_size = 50000  # Process in large chunks
    current_batch = []
    total_checked = 0

    print(f"\nChecking all {len(reactions)} reactions for token limits...")

    for r, g in zip(reactions, groups):
        text = f'Reaction:"{r}" Group:"{g}"'
        current_batch.append(text)

        if len(current_batch) >= batch_size:
            # Access underlying tokenizers library object for batch encoding
            batch_encodings = tokenizer.tokenizer.encode_batch(current_batch)

            for i, enc in enumerate(batch_encodings):
                if len(enc.ids) >= CONTEXT_WINDOW:
                    pytest.fail(
                        f"Entry exceeds limit! Tokens: {len(enc.ids)}. Content: {current_batch[i]}"
                    )

            total_checked += len(current_batch)
            print(f"Checked {total_checked}...", end="\r")
            current_batch = []

    # Check remaining
    if current_batch:
        batch_encodings = tokenizer.tokenizer.encode_batch(current_batch)
        for i, enc in enumerate(batch_encodings):
            if len(enc.ids) >= CONTEXT_WINDOW:
                pytest.fail(
                    f"Entry exceeds limit! Tokens: {len(enc.ids)}. Content: {current_batch[i]}"
                )
        total_checked += len(current_batch)

    print(f"\nSuccess: All {total_checked} items are within {CONTEXT_WINDOW} tokens.")
