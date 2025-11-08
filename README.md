# Hindi BPE Tokenizer

A Byte Pair Encoding (BPE) tokenizer specifically designed for Hindi text, using grapheme-level tokenization to properly handle Devanagari script.

## Overview

This tokenizer implements the BPE algorithm at the grapheme level rather than byte level, which is crucial for Hindi because:
- Hindi uses combining characters (matras) that attach to consonants
- Proper handling of conjuncts (consonant clusters)
- Preserves linguistic meaning better than byte-level tokenization

## Installation

### Requirements

```bash
pip install regex
```

### Setup

Clone the repository and ensure the `regex` package is installed:

```bash
git clone <repository-url>
cd hindi-bpe-tokenizer
pip install regex
```

## Quick Start

```python
from hindi_tokenizer import HindiTokenizer

# Initialize and train
tokenizer = HindiTokenizer()
tokenizer.train("आपका हिंदी टेक्स्ट यहाँ", num_merges=2000)

# Encode text
text = "मार्केट ट्रेंड्स को समझना"
encoded = tokenizer.encode(text)
print(f"Token IDs: {encoded}")

# Decode back
decoded = tokenizer.decode(encoded)
print(f"Decoded: {decoded}")

# Get compression stats
stats = tokenizer.get_compression_stats(text)
print(f"Compression: {stats['byte_compression_ratio']:.2f}x")
```

## Project Structure

```
hindi-bpe-tokenizer/
├── src/
│   └── hindi_tokenizer/
│       ├── __init__.py          # Package exports
│       ├── tokenizer.py         # Main HindiTokenizer class
│       ├── bpe.py              # Core BPE algorithm (get_stats, merge, run_merges)
│       └── grapheme.py         # Grapheme extraction for Hindi text
├── examples/
│   ├── basic_usage.py          # Simple usage example
│   └── advanced_training.py    # Training with larger corpus
├── tests/                      # Test directory
├── hindi_tokenization.py       # Original monolithic implementation
├── README.md
└── WARP.md                     # Guidance for Warp AI
```

## Usage Examples

### Basic Training and Encoding

```bash
python examples/basic_usage.py
```

### Advanced Training with Compression Analysis

```bash
python examples/advanced_training.py
```

## How It Works

### 1. Grapheme Extraction
The tokenizer extracts grapheme clusters (meaningful character units) from Hindi text using Unicode normalization:

```python
from hindi_tokenizer import graphemes

text = "शेयर बाज़ार"
tokens = graphemes(text)
# ['श', 'े', 'य', 'र', ' ', 'ब', 'ा', 'ज़', 'ा', 'र']
```

### 2. BPE Training
The algorithm iteratively merges the most frequent token pairs:

1. Start with individual graphemes as base vocabulary
2. Count frequencies of consecutive token pairs
3. Merge the most frequent pair into a new token
4. Repeat for N iterations (typically 2000-5000)

### 3. Encoding & Decoding
- **Encode**: Apply learned merge rules to compress text into token IDs
- **Decode**: Convert token IDs back to original text using vocabulary

## Key Features

- **Grapheme-aware**: Properly handles Hindi matras, conjuncts, and combining characters
- **Configurable compression**: Adjust `num_merges` parameter for vocabulary size vs compression tradeoff
- **Compression metrics**: Built-in statistics for analyzing tokenization efficiency
- **Type hints**: Full type annotations for better IDE support

## API Reference

### `HindiTokenizer`

Main tokenizer class with the following methods:

- `train(text: str, num_merges: int = 2000)` - Train on Hindi text
- `encode(text: str) -> List[int]` - Encode text to token IDs
- `decode(ids: List[int]) -> str` - Decode token IDs to text
- `get_vocab_size() -> int` - Get vocabulary size
- `get_compression_stats(text: str) -> Dict` - Get compression metrics

### Utility Functions

- `graphemes(text: str) -> List[str]` - Extract grapheme clusters
- `get_stats(ids: List[int]) -> Dict` - Count token pair frequencies
- `merge(ids: List[int], pair: Tuple, idx: int) -> List[int]` - Merge token pair
- `run_merges(num_merges: int, tokens: List[int], start_from_id: int)` - Execute BPE iterations

## Performance

Typical compression ratios with 2000 merges:
- **Grapheme compression**: 2-3x
- **Byte compression**: 2-4x

Higher merge counts increase vocabulary size but improve compression.

## License

[Specify your license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
