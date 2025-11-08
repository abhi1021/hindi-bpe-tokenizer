# -*- coding: utf-8 -*-
"""
Core Byte Pair Encoding (BPE) algorithm implementation.

This module contains the fundamental BPE operations:
- Counting token pair frequencies
- Merging token pairs
- Running multiple merge iterations
"""

from typing import Dict, List, Tuple


def get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """
    Count frequencies of consecutive token pairs.
    
    Args:
        ids: List of token IDs
        
    Returns:
        Dictionary mapping token pairs to their occurrence count
    """
    counts = {}
    for pair in zip(ids, ids[1:]):  # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """
    Replace all consecutive occurrences of a token pair with a new token.
    
    Args:
        ids: List of token IDs
        pair: Tuple of two token IDs to merge
        idx: New token ID to replace the pair with
        
    Returns:
        New list with all occurrences of pair replaced by idx
    """
    newids = []
    i = 0
    while i < len(ids):
        # If we are not at the very last position AND the pair matches, replace it
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def run_merges(
    num_merges: int,
    tokens: List[int],
    start_from_id: int
) -> Tuple[Dict[Tuple[int, int], int], List[int]]:
    """
    Execute multiple BPE merge iterations.
    
    In each iteration:
    1. Count all token pair frequencies
    2. Find the most frequent pair
    3. Merge all occurrences of that pair into a new token
    4. Record the merge rule
    
    Args:
        num_merges: Number of merge iterations to perform
        tokens: Initial list of token IDs
        start_from_id: Starting ID for newly created merged tokens
        
    Returns:
        Tuple of (merge_rules, compressed_tokens)
        - merge_rules: Dictionary mapping (token1, token2) -> new_token_id
        - compressed_tokens: Final compressed token sequence
    """
    merges = {}  # (int, int) -> int
    new_tokens = list(tokens)  # Copy to avoid modifying original
    idx = start_from_id
    
    for i in range(num_merges):
        stats = get_stats(new_tokens)
        if not stats:
            # No more pairs to merge
            break
        
        pair = max(stats, key=stats.get)
        new_tokens = merge(new_tokens, pair, idx)
        merges[pair] = idx
        idx += 1
    
    return merges, new_tokens
