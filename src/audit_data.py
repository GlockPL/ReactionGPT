import pickle
import random
import sys
from collections import Counter
from pathlib import Path

# Add project root to sys.path
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
sys.path.append(str(src_path := project_root / "src"))

from src.Model.Tokenizer import ChemTokenizer


def audit():
    print(f"{50 * '='}\nDATA AUDIT\n{50 * '='}")

    # 1. Load Data
    data_path = project_root / "data" / "pretraining_data.pickle"
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    reactions = data["reactions"]
    groups = data["groups"]

    total = len(reactions)
    print(f"Total entries: {total}")

    # 2. Check for Duplicates / Repetition
    print("\nChecking for duplicates in first 10,000 entries...")
    subset_r = reactions[:10000]
    unique_r = set(subset_r)
    print(f"Unique reactions in first 10k: {len(unique_r)}")

    if len(unique_r) < 100:
        print("CRITICAL WARNING: Data seems highly repetitive!")
        print(f"Sample unique: {list(unique_r)[:3]}")

    # 3. Check Tokenizer Output
    tokenizer_path = project_root / "ChemBPETokenizer" / "tokenizer.json"
    tokenizer = ChemTokenizer(tokenizer_path=str(tokenizer_path))

    print("\nChecking Tokenization of 5 random samples...")
    all_tokens = []

    for _ in range(5):
        idx = random.randint(0, total - 1)
        text = f'Reaction:"{reactions[idx]}" Group:"{groups[idx]}"'
        encoded = tokenizer.encode(text)

        print(f"\n--- Sample {idx} ---")
        print(f"Text length: {len(text)}")
        print(f"Text snippet: {text[:100]}...")
        print(f"Token IDs: {encoded.ids[:20]}...")
        print(f"Num Tokens: {len(encoded.ids)}")

        all_tokens.extend(encoded.ids)

    # 4. Check Token Distribution
    print(f"\n{50 * '='}\nTOKEN DISTRIBUTION (Sample)\n{50 * '='}")
    ctr = Counter(all_tokens)
    print(f"Most common tokens: {ctr.most_common(10)}")

    vocab_size = tokenizer.vocab_size
    print(f"Vocab Size: {vocab_size}")

    if len(ctr) < 10:
        print("CRITICAL WARNING: Model is using very few unique tokens.")

    # 5. Check Train/Val Split logic
    print(f"\n{50 * '='}\nSPLIT CHECK\n{50 * '='}")
    train_ratio = 0.90
    split_idx = int(train_ratio * total)

    train_sample = reactions[0]
    val_sample = reactions[split_idx]

    print(f"Split Index: {split_idx}")
    print(f"Train Sample [0]: {train_sample[:50]}...")
    print(f"Val Sample [{split_idx}]: {val_sample[:50]}...")

    if train_sample == val_sample:
        print("WARNING: Train[0] and Val[0] are identical. Check data sorting!")

    # 6. Check Dataloader Output
    print(f"\n{50 * '='}\nDATALOADER CHECK\n{50 * '='}")
    from src.Model.GPTDataloader import create_dataloader_v1

    # Create a small dataloader with just a few samples
    small_reactions = reactions[:1000]
    small_groups = groups[:1000]

    # Config for dataloader
    batch_size = 2
    context_length = 256
    stride = 128

    print("Creating temporary dataloader...")
    dataloader = create_dataloader_v1(
        small_reactions,
        small_groups,
        batch_size=batch_size,
        max_length=context_length,
        stride=stride,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    print("Iterating over first batch...")
    try:
        # Get first batch
        input_batch, target_batch = next(iter(dataloader))

        print(f"Input batch shape: {input_batch.shape}")
        print(f"Target batch shape: {target_batch.shape}")

        # Decode first sample in batch
        print("\n--- Decoded Batch 0 Sample 0 ---")
        input_ids = input_batch[0].tolist()
        target_ids = target_batch[0].tolist()

        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
        decoded_target = tokenizer.decode(target_ids, skip_special_tokens=False)

        print(f"Input Text:\n{decoded_input}")
        print(f"\nTarget Text:\n{decoded_target}")

        # Check if target is shifted by 1
        print("\n--- Shift Check ---")
        print(f"Input IDs  [:10]: {input_ids[:10]}")
        print(f"Target IDs [:10]: {target_ids[:10]}")

        if input_ids[1:] == target_ids[:-1]:
            print("SUCCESS: Target is correctly shifted by 1 position.")
        else:
            print("WARNING: Target is NOT shifted correctly!")
    except StopIteration:
        print("Dataloader returned no batches! Adjust context_length or stride.")


if __name__ == "__main__":
    audit()
