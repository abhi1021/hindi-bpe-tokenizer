# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a Hindi Byte Pair Encoding (BPE) tokenizer implementation that converts Hindi text into token sequences. The project demonstrates the complete BPE algorithm applied to Hindi language, including:

- Grapheme extraction from Unicode text using NFC normalization
- Building Hindi-specific vocabulary from grapheme combinations (consonants, vowels, matras, conjuncts)
- Iterative merging of most-frequent token pairs to create subword units
- Text encoding/decoding using learned merge rules

## Project Structure

```
hindi-bpe-tokenizer/
├── src/hindi_tokenizer/      # Main package
│   ├── __init__.py          # Package exports
│   ├── tokenizer.py         # HindiTokenizer class (main API)
│   ├── bpe.py              # Core BPE algorithm functions
│   └── grapheme.py         # Hindi grapheme extraction
├── src/                # Usage examples
│   ├── basic_usage.py      # Simple demo
│   └── advanced_training.py # Training analysis
├── tests/                   # Test directory
└── hindi_tokenization.py    # Original monolithic version
```

## Commands

### Run Examples

```bash
# Basic usage demonstration
python src/basic_usage.py

# Advanced training with compression analysis
python src/advanced_training.py
```

### Using the Package

```bash
# Install dependencies
pip install -r requirements.txt

# Use in Python code
python -c "from src.hindi_tokenizer import HindiTokenizer; print('Import successful')"
```

## Architecture and Key Concepts

### Grapheme-Based Tokenization
The tokenizer works at the grapheme level (meaningful Unicode characters) rather than raw UTF-8 bytes. This is crucial for Hindi because:
- Hindi uses combining characters (matras) that attach to consonants
- The `graphemes()` function uses regex `\X` with NFC normalization to extract proper grapheme clusters
- This preserves linguistic meaning better than byte-level tokenization

### Hindi Linguistic Components
The implementation recognizes:
- **Independent vowels** (स्वर): अ, आ, इ, ई, आदि
- **Consonants** (व्यंजन): क, ख, ग, घ, आदि
- **Matras** (vowel marks): ि, ी, ु, ू, आदि (attached to consonants)
- **Conjuncts** (consonant clusters): Consonant + Virama (्) + Consonant
- **Nasals/Visarga** (अनुस्वार/विसर्ग): ं, ः, ँ

### BPE Algorithm Flow
1. **Initialization**: Extract all unique graphemes from text
2. **Statistics**: Count consecutive token pair frequencies
3. **Merging**: Repeatedly merge the most frequent pair, creating new tokens
4. **Compression**: After N merges, sequences compress by replacing pairs with single tokens

### Module Organization

**`tokenizer.py`** - Main API (HindiTokenizer class)
- `train(text, num_merges)` - Train tokenizer on Hindi text
- `encode(text)` - Encode text to token IDs using learned merges
- `decode(ids)` - Convert token IDs back to text
- `get_vocab_size()` - Return vocabulary size
- `get_compression_stats(text)` - Calculate compression metrics

**`bpe.py`** - Core BPE algorithm
- `get_stats(ids)` - Count consecutive token pair frequencies
- `merge(ids, pair, idx)` - Replace all occurrences of a token pair with a new token
- `run_merges(num_merges, tokens, start_from_id)` - Execute multiple merge iterations

**`grapheme.py`** - Hindi text processing
- `graphemes(text)` - Extract grapheme clusters using Unicode regex
- `build_hindi_grapheme_set()` - Generate all possible Hindi grapheme combinations

## Workflow

The typical workflow for the tokenizer:

1. **Extract graphemes** from raw Hindi text
2. **Build initial vocabulary** from unique graphemes
3. **Create ID mappings** (token to ID, ID to token)
4. **Run BPE merges** (typically 2000-5000 iterations) to create vocabulary
5. **Encode new text** using the learned merge rules
6. **Decode** token sequences back to readable Hindi

## Key Parameters

- **num_merges**: Number of BPE iterations (default 2000). Higher values create larger vocabularies with better compression
- **startFromId**: ID offset for new merged tokens (typically set to initial vocabulary size)

## Compression Metrics

The tokenizer measures compression by comparing:
- Original raw UTF-8 bytes
- Grapheme-level tokens
- BPE-compressed tokens

Typical results show 2-4X compression ratios depending on text length and merge iterations.
