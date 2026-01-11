from src.Model.Tokenizer import ChemTokenizer
from src.utils import get_project_root


def get_config():
    # Robustly find the tokenizer path
    # Assuming this file is in src/config.py, project root is one level up
    project_root = get_project_root()
    tokenizer_path = project_root / "ChemBPETokenizer" / "tokenizer.json"

    tokenizer = ChemTokenizer(tokenizer_path=str(tokenizer_path))

    return {
        "vocab_size": tokenizer.vocab_size,  # Vocabulary size
        "context_length": 2024,  # Context length
        "emb_dim": 1024,  # Embedding dimension
        "n_heads": 16,  # Number of attention heads
        "n_layers": 28,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-Key-Value bias
    }
