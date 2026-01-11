from pathlib import Path

import torch

from config import get_config
from src.Model.GPT import GPTModel, generate_text_simple
from src.Model.Tokenizer import ChemTokenizer
from src.utils import load_latest_checkpoint


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    gpt_config = get_config()
    file_path = Path(__file__).resolve()
    project_root = file_path.parent.parent
    checkpoint_dir = project_root / "checkpoints"

    # Initialize Model
    model = GPTModel(gpt_config)
    model.to(device)

    # Load Checkpoint
    epoch = load_latest_checkpoint(model, checkpoint_dir)
    if epoch is None:
        print("Failed to load model. Exiting.")
        return
    print(f"Model loaded from epoch {epoch}")

    # Initialize Tokenizer
    tokenizer_path = project_root / "ChemBPETokenizer" / "tokenizer.json"
    tokenizer = ChemTokenizer(tokenizer_path=str(tokenizer_path))

    # Input Reaction
    # input_reaction = "ClS(C1=CC=CC=C1)(=O)=O.C=CC[B-](F)(F)F>>O=S(C2=CC=CC=C2)(CC=C)=O"
    input_reaction = "COC(C)=O.NC(=O)n1ncc(-c2ccccc2)c1N>>Oc1ncnc2c(-c3ccccc3)cnn12"

    # Format prompt to match training data: Reaction:"..." Group:"
    # We want the model to complete the Group part.
    prompt = f'Reaction:"{input_reaction}" Group:"'

    print(f"\n{50 * '='}")
    print("INPUT PROMPT:")
    print(prompt)
    print(f"{50 * '='}")

    # Encode
    encoded = tokenizer.encode(prompt)
    encoded_tensor = torch.tensor(encoded.ids).unsqueeze(0).to(device)

    # Generate
    print("\nGenerating...")
    with torch.no_grad():
        # Generate enough tokens to cover a typical group SMILES
        out_ids = generate_text_simple(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=50,
            context_size=gpt_config["context_length"],
        )

    # Decode
    decoded_text = tokenizer.decode(
        out_ids.squeeze(0).tolist(), skip_special_tokens=True
    )

    # The output includes the prompt. Let's extract just the generation if possible,
    # or just print the whole thing.

    print(f"\n{50 * '='}")
    print("FULL OUTPUT:")
    print(decoded_text)
    print(f"{50 * '='}")

    # Try to extract just the generated group
    if prompt in decoded_text:
        generated_part = decoded_text[len(prompt) :]
        # It might end with a closing quote if the model learned that
        print("\nGenerated Group (Parsed):")
        print(generated_part.split('"')[0])


if __name__ == "__main__":
    main()
