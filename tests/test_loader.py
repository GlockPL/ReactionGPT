from src.Model.ClassificationDataLoader import ClassificationDataset
from src.Model.Tokenizer import ChemTokenizer
from src.utils import get_project_root


def test_spam_dataset():
    print("Testing ClassificationDataset...")

    # Path to the balanced dataset
    data_path = get_project_root() / "data" / "balanced_classification_data.pickle"
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        return

    # Initialize Tokenizer
    tokenizer_path = get_project_root() / "ChemBPETokenizer" / "tokenizer.json"
    tokenizer = ChemTokenizer(tokenizer_path=str(tokenizer_path))

    # Initialize Dataset
    dataset = ClassificationDataset(
        pickle_file_path=str(data_path), tokenizer=tokenizer, max_length=128
    )

    print(f"Dataset length: {len(dataset)}")

    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    # Test __getitem__
    item_idx = 0
    input_ids, label = dataset[item_idx]

    print(f"Item {item_idx}:")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Input IDs (first 10): {input_ids[:10]}")
    print(f"Label: {label}")

    # Check padding
    assert len(input_ids) == 128, f"Expected length 128, got {len(input_ids)}"

    print("SpamDataset test passed!")


if __name__ == "__main__":
    test_spam_dataset()
