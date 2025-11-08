# -*- coding: utf-8 -*-
"""
Hindi BPE Tokenizer Package

A Byte Pair Encoding tokenizer specifically designed for Hindi text,
using grapheme-level tokenization to properly handle Devanagari script.
"""

from .tokenizer import HindiTokenizer
from .grapheme import graphemes, build_hindi_grapheme_set
from .bpe import get_stats, merge, run_merges

__version__ = "0.1.0"

__all__ = [
    "HindiTokenizer",
    "graphemes",
    "build_hindi_grapheme_set",
    "get_stats",
    "merge",
    "run_merges",
]
