#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload Hindi BPE Tokenizer to HuggingFace Hub.

This script trains the tokenizer and uploads it to HuggingFace Hub along with
model card and configuration files.

Usage:
    python upload_to_huggingface.py --repo-name your-username/hindi-bpe-tokenizer

Requirements:
    pip install huggingface-hub
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hindi_tokenizer import HindiTokenizer


def create_model_card(stats: dict, num_merges: int) -> str:
    """Create a model card in markdown format."""
    return f"""---
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

- **Training corpus size**: {stats['utf8_bytes']:,} UTF-8 bytes
- **Number of merges**: {num_merges:,}
- **Vocabulary size**: {stats.get('vocab_size', 'N/A'):,}
- **Grapheme compression ratio**: {stats['grapheme_compression_ratio']:.2f}x
- **Byte compression ratio**: {stats['byte_compression_ratio']:.2f}x

## Performance

### Compression Ratios

The tokenizer achieves excellent compression while maintaining perfect reversibility:

- **Raw graphemes**: {stats['raw_graphemes']:,}
- **Compressed tokens**: {stats['compressed_tokens']:,}
- **Grapheme compression**: {stats['grapheme_compression_ratio']:.2f}x
- **Byte compression**: {stats['byte_compression_ratio']:.2f}x

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
print(f"Token IDs: {{encoded}}")

# Decode back
decoded = tokenizer.decode(encoded)
print(f"Decoded: {{decoded}}")

# Get compression stats
stats = tokenizer.get_compression_stats(text)
print(f"Compression: {{stats['byte_compression_ratio']:.2f}}x")
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
@software{{hindi_bpe_tokenizer,
  title = {{Hindi BPE Tokenizer}},
  author = {{Your Name}},
  year = {{2025}},
  url = {{https://github.com/your-username/hindi-bpe-tokenizer}}
}}
```

## Repository

[GitHub Repository](https://github.com/your-username/hindi-bpe-tokenizer)
"""


def create_tokenizer_config(tokenizer: HindiTokenizer, num_merges: int) -> dict:
    """Create tokenizer configuration."""
    return {
        "tokenizer_class": "HindiTokenizer",
        "model_type": "BPE",
        "num_merges": num_merges,
        "vocab_size": tokenizer.get_vocab_size(),
        "grapheme_based": True,
        "language": "hindi",
        "script": "devanagari"
    }


def save_tokenizer_artifacts(tokenizer: HindiTokenizer, output_dir: Path,
                            num_merges: int, stats: dict):
    """Save tokenizer files for upload."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save tokenizer as pickle
    import pickle
    tokenizer_path = output_dir / "hindi_tokenizer.pkl"
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"✓ Saved tokenizer to {tokenizer_path}")

    # Save vocabulary
    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer.vocab, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved vocabulary to {vocab_path}")

    # Save merges
    merges_path = output_dir / "merges.json"
    # Convert tuple keys to strings for JSON serialization
    merges_serializable = {f"{k[0]},{k[1]}": v for k, v in tokenizer.merges.items()}
    with open(merges_path, 'w', encoding='utf-8') as f:
        json.dump(merges_serializable, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved merges to {merges_path}")

    # Save config
    config_path = output_dir / "tokenizer_config.json"
    config = create_tokenizer_config(tokenizer, num_merges)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved config to {config_path}")

    # Save model card
    stats_with_vocab = {**stats, 'vocab_size': tokenizer.get_vocab_size()}
    model_card = create_model_card(stats_with_vocab, num_merges)
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(model_card)
    print(f"✓ Saved model card to {readme_path}")


def upload_to_hub(output_dir: Path, repo_id: str, token: str = None):
    """Upload tokenizer to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, create_repo

        api = HfApi(token=token)

        # Create repository (will not fail if exists)
        print(f"\nCreating repository: {repo_id}")
        create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
        print(f"✓ Repository ready: https://huggingface.co/{repo_id}")

        # Upload all files
        print("\nUploading files...")
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=repo_id,
            token=token,
            commit_message="Upload Hindi BPE Tokenizer"
        )

        print(f"\n✅ Successfully uploaded tokenizer to https://huggingface.co/{repo_id}")

    except ImportError:
        print("\n❌ Error: huggingface-hub package not installed")
        print("Install it with: pip install huggingface-hub")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error uploading to HuggingFace Hub: {e}")
        print("\nMake sure you:")
        print("1. Have a HuggingFace account")
        print("2. Are logged in: huggingface-cli login")
        print("3. Or provide token via --token argument")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Train and upload Hindi BPE Tokenizer to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="HuggingFace repository name (e.g., username/hindi-bpe-tokenizer)"
    )
    parser.add_argument(
        "--num-merges",
        type=int,
        default=4000,
        help="Number of BPE merges (default: 4000)"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="src/content.txt",
        help="Path to training corpus (default: src/content.txt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tokenizer_files",
        help="Directory to save tokenizer files (default: tokenizer_files)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (optional, will use huggingface-cli login if not provided)"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Save files locally without uploading to HuggingFace"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Hindi BPE Tokenizer - HuggingFace Upload")
    print("=" * 70)

    # Load training corpus
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"❌ Error: Training corpus not found at {corpus_path}")
        sys.exit(1)

    print(f"\n📖 Loading training corpus from {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        training_corpus = f.read()
    print(f"✓ Loaded {len(training_corpus):,} characters")

    # Train tokenizer
    print(f"\n🔧 Training tokenizer with {args.num_merges:,} merges...")
    tokenizer = HindiTokenizer()
    tokenizer.train(training_corpus, num_merges=args.num_merges)
    print(f"✓ Training complete")
    print(f"  Vocabulary size: {tokenizer.get_vocab_size():,}")

    # Get compression stats
    print("\n📊 Computing compression statistics...")
    stats = tokenizer.get_compression_stats(training_corpus)
    print(f"✓ Compression analysis complete")
    print(f"  UTF-8 bytes: {stats['utf8_bytes']:,}")
    print(f"  Raw graphemes: {stats['raw_graphemes']:,}")
    print(f"  Compressed tokens: {stats['compressed_tokens']:,}")
    print(f"  Grapheme compression: {stats['grapheme_compression_ratio']:.2f}x")
    print(f"  Byte compression: {stats['byte_compression_ratio']:.2f}x")

    # Save artifacts
    output_dir = Path(args.output_dir)
    print(f"\n💾 Saving tokenizer files to {output_dir}/")
    save_tokenizer_artifacts(tokenizer, output_dir, args.num_merges, stats)

    # Upload to HuggingFace Hub
    if not args.no_upload:
        print("\n" + "=" * 70)
        upload_to_hub(output_dir, args.repo_name, args.token)
    else:
        print(f"\n✅ Tokenizer files saved locally to {output_dir}/")
        print(f"\nTo upload later, run:")
        print(f"  python {__file__} --repo-name {args.repo_name} --no-training")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()