#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced training example for Hindi BPE Tokenizer.

This script demonstrates training with a larger corpus and analyzing
the results with different merge counts.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hindi_tokenizer import HindiTokenizer

# Load the training corpus from content.txt
with open('content.txt', 'r', encoding='utf-8') as f:
    TRAINING_CORPUS = f.read()


def main():
    print("=" * 70)
    print("Hindi BPE Tokenizer - Advanced Training Example")
    print("=" * 70)
    
    # Test with different merge counts
    merge_counts = [100, 500, 1000, 2000]
    
    test_text = "इन ट्रेंड्स को जल्दी पहचानने से ट्रेडर्स को मदद मिलती है"
    
    print(f"\nTest text: '{test_text}'")
    print(f"UTF-8 bytes: {len(test_text.encode('utf-8'))}")
    
    print("\n" + "-" * 70)
    print(f"{'Merges':<10} {'Vocab Size':<15} {'Tokens':<10} {'Compression':<15}")
    print("-" * 70)
    
    for num_merges in merge_counts:
        # Train tokenizer
        tokenizer = HindiTokenizer()
        tokenizer.train(TRAINING_CORPUS, num_merges=num_merges)
        
        # Get stats
        encoded = tokenizer.encode(test_text)
        stats = tokenizer.get_compression_stats(test_text)
        
        print(f"{num_merges:<10} {tokenizer.get_vocab_size():<15} "
              f"{len(encoded):<10} {stats['byte_compression_ratio']:.2f}x")
    
    print("-" * 70)
    
    # Detailed analysis with optimal merge count
    print("\n\nDetailed analysis with 2000 merges:")
    print("=" * 70)
    
    tokenizer = HindiTokenizer()
    tokenizer.train(TRAINING_CORPUS, num_merges=4000)
    
    # Encode and decode
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nOriginal:  '{test_text}'")
    print(f"Decoded:   '{decoded}'")
    print(f"Match:     {test_text == decoded}")
    
    print(f"\nToken IDs: {encoded}")
    print(f"\nToken breakdown:")
    for idx, token_id in enumerate(encoded):
        token_str = tokenizer.vocab[token_id]
        print(f"  Position {idx}: ID={token_id:4d} → '{token_str}'")
    
    # Overall compression stats
    stats = tokenizer.get_compression_stats(TRAINING_CORPUS)
    print("\n\nTraining corpus compression:")
    print(f"  UTF-8 bytes:              {stats['utf8_bytes']:,}")
    print(f"  Raw graphemes:            {stats['raw_graphemes']:,}")
    print(f"  Compressed tokens:        {stats['compressed_tokens']:,}")
    print(f"  Grapheme compression:     {stats['grapheme_compression_ratio']:.2f}x")
    print(f"  Byte compression:         {stats['byte_compression_ratio']:.2f}x")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
