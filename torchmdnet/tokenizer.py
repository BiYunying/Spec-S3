"""Single SMILES decoder vocabulary shared by pretraining and downstream tasks."""

from __future__ import annotations

SMILES_DICTIONARY = {
    "PAD": 0,
    "BOS": 1,
    "EOS": 2,
    "#": 3,
    "(": 4,
    ")": 5,
    "1": 6,
    "2": 7,
    "3": 8,
    "4": 9,
    "5": 10,
    "6": 11,
    "7": 12,
    "8": 13,
    "=": 14,
    "Br": 15,
    "C": 16,
    "Cl": 17,
    "F": 18,
    "I": 19,
    "N": 20,
    "O": 21,
    "P": 22,
    "S": 23,
    "[CH-]": 24,
    "[CH2-]": 25,
    "[N+]": 26,
    "[N-]": 27,
    "[NH+]": 28,
    "[NH-]": 29,
    "[NH2+]": 30,
    "[NH3+]": 31,
    "[O+]": 32,
    "[O-]": 33,
    "[OH+]": 34,
    "[P+]": 35,
    "[PH+]": 36,
    "[PH2]": 37,
    "[PH]": 38,
    "[S+]": 39,
    "[S-]": 40,
    "[SH+]": 41,
    "[C-]": 42,
    "[Si]": 43,
    "[SiH3]": 44,
    "[SiH2]": 45,
    "[CH]": 46,
    "[SiH]": 47,
    "[CH2+]": 48,
    "[C]": 49,
    "[C+]": 50,
    "[CH+]": 51,
    "[N]": 52,
    "[SH]": 53,
    "[O]": 54,
    "[CH2]": 55,
    "[PH3+]": 56,
    "[Cl+3]": 57,
    "[P-]": 58,
    "[PH2+]": 59,
    "[SiH-]": 60,
    "[OH2+]": 61,
    "[S]": 62,
    "[Si+]": 63,
    "[PH-]": 64,
    "[Si-]": 65,
}
# Keep the historical import name as an alias, not a second definition.
FG26_SMILES_DICTIONARY = SMILES_DICTIONARY


def get_smiles_dictionary(vocab=None, *, vocab_size=None):
    """Resolve and validate token IDs, including the 43-token checkpoint prefix.

    New models use all 66 tokens. Only checkpoint loading supplies vocab_size;
    historical 43-token decoder rows keep their original meanings.
    """
    if vocab is None:
        if vocab_size not in (None, 43, len(SMILES_DICTIONARY)):
            raise ValueError("Unknown decoder vocabulary size; checkpoint must supply its exact smiles_dictionary")
        vocab = {token: index for token, index in SMILES_DICTIONARY.items()
                 if vocab_size is None or index < vocab_size}
    else:
        vocab = dict(vocab)
    if any(not isinstance(token, str) or type(index) is not int for token, index in vocab.items()):
        raise ValueError("SMILES vocabulary must map string tokens to integer IDs")
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("SMILES vocabulary IDs must be unique and contiguous from zero")
    if any(vocab.get(token) != index for index, token in enumerate(("PAD", "BOS", "EOS"))):
        raise ValueError("SMILES vocabulary requires PAD=0, BOS=1, EOS=2")
    if vocab_size is not None and len(vocab) != vocab_size:
        raise ValueError("SMILES vocabulary size differs from the decoder checkpoint")
    return vocab


get_smiles_dictionary(SMILES_DICTIONARY)

import re
import torch
from typing import List, Dict


class CustomTokenizer:

    def __init__(self, vocab_dict: dict, add_missing_special_tokens: bool = True):
        """
        Initialize from a Python vocabulary dictionary.
        """
        self.vocab = get_smiles_dictionary(vocab_dict)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.bos_token = "BOS"
        self.eos_token = "EOS"
        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab["PAD"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        """
        Split a SMILES string into tokens.
        The regular expression preserves bracketed tokens such as [N+] and [O-].
        """
        pattern = "(\\[[^\\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\\(|\\)|\\.|=|#|-|\\+|\\\\|\\/|:|~|@|\\?|>|\\*|\\$|\\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(text)]
        return tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Map tokens to IDs, returning None for unknown tokens."""
        return [self.vocab.get(token) for token in tokens]

    def decode(self, ids: List[int]) -> str:
        special_ids = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        tokens = [
            self.inv_vocab.get(id)
            for id in ids
            if id not in special_ids and id in self.inv_vocab
        ]
        return "".join(tokens)

    def __call__(
        self,
        texts: List[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch with a Hugging Face-style interface.
        """
        batch_ids = []
        for text in texts:
            tokens = self._tokenize(text)
            if truncation and len(tokens) > max_length - 2:
                tokens = tokens[: max_length - 2]
            ids = (
                [self.bos_token_id]
                + self.convert_tokens_to_ids(tokens)
                + [self.eos_token_id]
            )
            batch_ids.append(ids)
        if padding:
            max_len_in_batch = max((len(ids) for ids in batch_ids))
            padded_batch_ids = []
            attention_masks = []
            for ids in batch_ids:
                padding_len = max_len_in_batch - len(ids)
                padded_ids = ids + [self.pad_token_id] * padding_len
                attention_mask = [1] * len(ids) + [0] * padding_len
                padded_batch_ids.append(padded_ids)
                attention_masks.append(attention_mask)
            return {
                "input_ids": torch.tensor(padded_batch_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }
        else:
            return {
                "input_ids": [torch.tensor(ids, dtype=torch.long) for ids in batch_ids]
            }

    def get_special_tokens_mask(self, token_ids_0: List[int]) -> List[int]:
        """
        Identify special tokens for masked language modeling.
        """
        special_tokens = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        return [1 if token_id in special_tokens else 0 for token_id in token_ids_0]


from collections.abc import Sequence
from rdkit import Chem


class CanonicalNonIsomericFG26Tokenizer(CustomTokenizer):
    """Strict targets; canonical by default, optional training enumeration."""

    def _normalize(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid FG26 SMILES target: {smiles!r}")
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        normalized = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=False, kekuleSmiles=True
        )
        tokens = self._tokenize(normalized)
        if "".join(tokens) != normalized:
            raise ValueError(
                f"FG26 target cannot be losslessly tokenized after normalization: {normalized!r}"
            )
        unsupported = sorted({token for token in tokens if token not in self.vocab})
        if unsupported:
            raise ValueError(
                f"FG26 target contains unsupported tokens {unsupported}: {normalized!r}"
            )
        return normalized

    def _randomize(self, smiles: str, max_tokens: int | None) -> str:
        # Match the original random-SMILES experiment: seeded PyTorch atom
        # permutations, non-isomeric Kekule output, and no silent truncation.
        mol = Chem.MolFromSmiles(self._normalize(smiles))
        identity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        for _ in range(64):
            permuted = Chem.RenumberAtoms(mol, torch.randperm(mol.GetNumAtoms()).tolist())
            candidate = Chem.MolToSmiles(
                permuted, canonical=False, isomericSmiles=False, kekuleSmiles=True
            )
            tokens = self._tokenize(candidate)
            if ("".join(tokens) != candidate
                    or any(token not in self.vocab for token in tokens)
                    or (max_tokens is not None and len(tokens) > max_tokens)):
                continue
            restored = Chem.MolFromSmiles(candidate)
            if restored is not None and Chem.MolToSmiles(
                restored, canonical=True, isomericSmiles=False
            ) == identity:
                return candidate
        raise ValueError(
            f"No encodable identity-preserving random SMILES after 64 attempts: {smiles!r}"
        )

    def __call__(
        self,
        texts: Sequence[str] | str,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        randomize: bool = False,
    ):
        if isinstance(texts, str):
            texts = [texts]
        normalized = [
            self._randomize(text, max_length - 2 if truncation else None)
            if randomize else self._normalize(text)
            for text in texts
        ]
        if truncation:
            too_long = [
                text
                for text in normalized
                if len(self._tokenize(text)) > max_length - 2
            ]
            if too_long:
                raise ValueError(
                    f"Refusing to truncate FG26 cross-attention decoder targets; first overlength target: {too_long[0]!r}"
                )
        return super().__call__(
            normalized, padding=padding, truncation=truncation, max_length=max_length
        )
