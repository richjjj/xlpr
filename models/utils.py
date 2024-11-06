from typing import List, Union
import torch
from torch import Tensor
from dataclasses import dataclass, field


@dataclass
class StringLabelConverter:
    """Convert between strings and labels for OCR tasks.

    This converter handles encoding text to tensor indices and decoding back to text,
    with support for CTC blank labels.

    Args:
        alphabet: Set of possible characters
        ignore_case: Whether to ignore character case
    """

    alphabet: str
    ignore_case: bool = False
    _char_to_idx: dict = field(init=False)

    def __post_init__(self):
        if self.ignore_case:
            self.alphabet = self.alphabet.lower()
        self.alphabet = "-" + self.alphabet

        # Create character to index mapping
        self._char_to_idx = {char: idx for idx, char in enumerate(self.alphabet)}

    def encode_list(self, texts: List[Union[str, bytes]], max_len: int = 8) -> Tensor:
        """Encode texts with padding to fixed length.

        Args:
            texts: List of texts to encode
            max_len: Maximum length to pad/truncate to

        Returns:
            Tensor of shape (batch_size, max_len) containing character indices
        """
        batch = []

        for text in texts:
            if isinstance(text, bytes):
                text = text.decode("utf-8")

            # Create fixed-length sequence padded with zeros
            indices = [self._char_to_idx[char] for char in text[:max_len]]
            indices.extend([0] * (max_len - len(indices)))
            batch.append(indices)

        return torch.tensor(batch, dtype=torch.int64)

    def decode_list(self, encoded: Tensor) -> List[str]:
        """Decode padded sequences back to strings.

        Args:
            encoded: Tensor of shape (batch_size, seq_len) containing indices

        Returns:
            List of decoded strings with padding removed
        """
        return [
            "".join(self.alphabet[idx] for idx in seq if idx != 0) for seq in encoded
        ]


@dataclass
class Averager:
    """Compute running average for PyTorch tensors and variables."""

    sum: float = 0.0
    count: int = 0

    def add(self, value: Union[Tensor, "torch.autograd.Variable"]):
        """Add a new value to the average."""
        if isinstance(value, torch.autograd.Variable):
            count = value.data.numel()
            value = value.data.sum()
        else:
            count = value.numel()
            value = value.sum()

        self.count += count
        self.sum += float(value)

    def reset(self):
        """Reset the averager."""
        self.sum = 0.0
        self.count = 0

    def value(self) -> float:
        """Get current average value."""
        return self.sum / self.count if self.count != 0 else 0.0
