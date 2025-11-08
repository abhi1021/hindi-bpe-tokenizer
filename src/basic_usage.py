#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic usage example for Hindi BPE Tokenizer.

This script demonstrates:
- Training a tokenizer on Hindi text
- Encoding new text
- Decoding token sequences
- Computing compression statistics
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hindi_tokenizer import HindiTokenizer


def main():
    # Sample Hindi text about stock market analysis
    training_text = """
    शेयर बाज़ार अनैलिसिस के तरीके 💹📈
    
    मार्केट ट्रेंड्स (market trends) को समझना
    मान लो कि आप एक नए शहर जैसे न्यूयॉर्क में बिना जीपीएस के गाड़ी चला रहे हो। 
    आप देखते हो कि आपके आगे ज्यादातर कारें एक चौराहे पर बाईं ओर मुड़ रही हैं। 
    आपको नहीं पता कि वे क्यों मुड़ रही हैं, लेकिन आप मान लेते हो कि उन्हें सबसे अच्छा रास्ता पता होगा।
    
    पैटर्न पर भरोसा करते हुए, आप भी उनके पीछे चलते हैं। इसी तरह, टेक्निकल एनालिसिस 
    (Technical Analysis - TA) स्टॉक प्राइस के समय के साथ बनते पैटर्न्स को फॉलो करने के बारे में है।
    """
    
    print("=" * 70)
    print("Hindi BPE Tokenizer - Basic Usage Example")
    print("=" * 70)
    
    # Initialize and train tokenizer
    print("\n1. Training tokenizer...")
    tokenizer = HindiTokenizer()
    tokenizer.train(training_text, num_merges=100)
    print(f"   ✓ Trained with vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Encode sample text
    test_text = "मार्केट ट्रेंड्स को समझना बहुत महत्वपूर्ण है"
    print(f"\n2. Encoding text: '{test_text}'")
    encoded = tokenizer.encode(test_text)
    print(f"   Token IDs: {encoded}")
    print(f"   Number of tokens: {len(encoded)}")
    
    # Decode back
    print("\n3. Decoding tokens...")
    decoded = tokenizer.decode(encoded)
    print(f"   Decoded text: '{decoded}'")
    print(f"   ✓ Match: {decoded == test_text}")
    
    # Show compression stats
    print("\n4. Compression Statistics:")
    stats = tokenizer.get_compression_stats(test_text)
    print(f"   UTF-8 bytes: {stats['utf8_bytes']}")
    print(f"   Raw graphemes: {stats['raw_graphemes']}")
    print(f"   Compressed tokens: {stats['compressed_tokens']}")
    print(f"   Grapheme compression ratio: {stats['grapheme_compression_ratio']:.2f}x")
    print(f"   Byte compression ratio: {stats['byte_compression_ratio']:.2f}x")
    
    # Show some vocabulary items
    print("\n5. Sample vocabulary items:")
    vocab_items = list(tokenizer.vocab.items())[:10]
    for idx, token in vocab_items:
        print(f"   ID {idx:3d}: '{token}'")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
