"""S3 tokenization with tiktoken (encoding configurable, default cl100k_base)."""

from functools import lru_cache

import tiktoken


class Tokenizer:
    def __init__(self, encoding_name: str):
        self.encoding_name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        # disallowed_special=() so literal "<|endoftext|>" in a document is plain text, not an error
        return self._enc.encode(text, disallowed_special=())

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[int]]:
        """Token ids plus the character offset where each token starts in ``text``."""
        tokens = self.encode(text)
        decoded, offsets = self._enc.decode_with_offsets(tokens)
        if decoded != text:  # cannot happen for valid str input; guard against silent drift
            raise ValueError("tokenizer round-trip mismatch")
        return tokens, offsets

    def preview(self, text: str, limit: int = 300) -> list[dict[str, int | str]]:
        """First ``limit`` tokens as ``{id, text}`` for the token-boundary view."""
        tokens = self.encode(text)[:limit]
        return [
            {"id": t, "text": self._enc.decode_single_token_bytes(t).decode("utf-8", "replace")}
            for t in tokens
        ]


@lru_cache
def get_tokenizer(encoding_name: str) -> Tokenizer:
    return Tokenizer(encoding_name)
