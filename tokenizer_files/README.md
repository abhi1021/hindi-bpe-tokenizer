---
language:
- hi
license: mit
tags:
- tokenizer
- bpe
- hindi
- devanagari
- nlp
pipeline_tag: tokenization
---

# Hindi BPE Tokenizer

A Byte Pair Encoding (BPE) tokenizer specifically designed for Hindi text, using grapheme-level tokenization to properly handle Devanagari script.

## Model Description

This tokenizer implements the BPE algorithm at the grapheme level rather than byte level, which is crucial for Hindi because:
- Hindi uses combining characters (matras) that attach to consonants
- Proper handling of conjuncts (consonant clusters)
- Preserves linguistic meaning better than byte-level tokenization

## Training Details

- **Training corpus size**: 60,565 UTF-8 bytes
- **Number of merges**: 4,000
- **Vocabulary size**: 4,629
- **Grapheme compression ratio**: 5.28x
- **Byte compression ratio**: 18.62x

## Performance

### Compression Ratios

The tokenizer achieves excellent compression while maintaining perfect reversibility:

- **Raw graphemes**: 17,179
- **Compressed tokens**: 3,253
- **Grapheme compression**: 5.28x
- **Byte compression**: 18.62x

## Usage

```python
from hindi_tokenizer import HindiTokenizer
import pickle

# Load the tokenizer
with open('hindi_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

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

## Installation

```bash
pip install regex
```

## Limitations and Bias

This tokenizer is trained on Hindi text and is optimized for Devanagari script. Performance may vary on:
- Mixed language text (Hindi-English code-switching)
- Other Indic languages using Devanagari script
- Informal or social media text with non-standard spellings

## Citation

If you use this tokenizer in your research, please cite:

```bibtex
@software{hindi_bpe_tokenizer,
  title = {Hindi BPE Tokenizer},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-username/hindi-bpe-tokenizer}
}
```

## Repository

[GitHub Repository](https://github.com/your-username/hindi-bpe-tokenizer)
