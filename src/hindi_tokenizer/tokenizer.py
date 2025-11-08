# -*- coding: utf-8 -*-
"""
Hindi BPE Tokenizer with training and encoding capabilities.

This module provides a Tokenizer class that orchestrates the complete
BPE tokenization workflow for Hindi text.
"""

from typing import Dict, List, Optional
from .grapheme import graphemes
from .bpe import get_stats, merge, run_merges


class HindiTokenizer:
    """
    Byte Pair Encoding tokenizer for Hindi text.
    
    This tokenizer uses grapheme-level tokenization combined with BPE
    to create efficient subword vocabularies for Hindi text.
    """
    
    def __init__(self):
        """Initialize an empty tokenizer."""
        self.vocab: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: Dict[tuple, int] = {}
        self.trained = False
    
    def train(self, text: str, num_merges: int = 2000) -> None:
        """
        Train the tokenizer on Hindi text.
        
        Steps:
        1. Extract graphemes from text
        2. Build initial vocabulary from unique graphemes
        3. Create token-to-ID mappings
        4. Run BPE merge iterations
        5. Build final vocabulary with merged tokens
        
        Args:
            text: Training text in Hindi
            num_merges: Number of BPE merge iterations (default: 2000)
        """
        # Extract graphemes and build initial vocabulary
        raw_tokens = graphemes(text)
        base_vocab = sorted(set(raw_tokens))
        
        # Create bidirectional mappings
        self.token_to_id = {tok: idx for idx, tok in enumerate(base_vocab)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        
        # Convert tokens to IDs
        tokens = self._tokens_to_ids(raw_tokens)
        
        # Run BPE merges
        self.merges, compressed_tokens = run_merges(
            num_merges=num_merges,
            tokens=tokens,
            start_from_id=len(base_vocab)
        )
        
        # Build final vocabulary including merged tokens
        self.vocab = {idx: base_vocab[idx] for idx in range(len(base_vocab))}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        
        self.trained = True
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs using learned BPE merges.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
            
        Raises:
            ValueError: If tokenizer hasn't been trained yet
        """
        if not self.trained:
            raise ValueError("Tokenizer must be trained before encoding. Call train() first.")
        
        # Extract graphemes and convert to IDs
        raw_tokens = graphemes(text)
        tokens = self._tokens_to_ids(raw_tokens)
        
        # Apply learned merges
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # Nothing else can be merged
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text string
            
        Raises:
            ValueError: If tokenizer hasn't been trained yet
        """
        if not self.trained:
            raise ValueError("Tokenizer must be trained before decoding. Call train() first.")
        
        return "".join(self.vocab[idx] for idx in ids)
    
    def _tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert token strings to IDs.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of token IDs
        """
        return [self.token_to_id[tok] for tok in tokens]
    
    def get_vocab_size(self) -> int:
        """
        Get the size of the vocabulary.
        
        Returns:
            Number of tokens in vocabulary
        """
        return len(self.vocab)
    
    def get_compression_stats(self, text: str) -> Dict[str, float]:
        """
        Calculate compression statistics for given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with compression metrics
        """
        if not self.trained:
            raise ValueError("Tokenizer must be trained before computing stats.")
        
        utf8_bytes = len(text.encode("utf-8"))
        raw_graphemes = len(graphemes(text))
        encoded = self.encode(text)
        compressed_tokens = len(encoded)
        
        return {
            "utf8_bytes": utf8_bytes,
            "raw_graphemes": raw_graphemes,
            "compressed_tokens": compressed_tokens,
            "grapheme_compression_ratio": raw_graphemes / compressed_tokens if compressed_tokens > 0 else 0,
            "byte_compression_ratio": utf8_bytes / compressed_tokens if compressed_tokens > 0 else 0
        }
