from tokenizers import Tokenizer


class ChemTokenizer:
    def __init__(self, tokenizer_path="ChemBPETokenizer/tokenizer.json"):
        """
        Initializes the ChemTokenizer by loading a pre-trained tokenizer from a JSON file.
        
        Args:
            tokenizer_path (str): Path to the tokenizer.json file.
        """
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def encode(self, text):
        """
        Encodes a string into tokens.
        
        Args:
            text (str): The input string (e.g., 'Reaction:"..." Group:"..."').
            
        Returns:
            Encoding: The encoding object containing ids, tokens, etc.
        """
        return self.tokenizer.encode(text)

    def decode(self, ids, skip_special_tokens=True):
        """
        Decodes a list of token IDs back into a string.
        
        Args:
            ids (List[int]): List of token IDs.
            skip_special_tokens (bool): Whether to remove special tokens during decoding.
            
        Returns:
            str: The decoded string.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self):
        """Returns the vocabulary size of the tokenizer."""
        return self.tokenizer.get_vocab_size()

    def save(self, path):
        """Saves the tokenizer to the specified path."""
        self.tokenizer.save(path)
