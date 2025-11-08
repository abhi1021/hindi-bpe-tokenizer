# -*- coding: utf-8 -*-
"""
Grapheme extraction for Hindi text using Unicode normalization.

This module handles the extraction of grapheme clusters from Hindi text,
which is crucial for proper tokenization of Devanagari script that uses
combining characters (matras) and complex conjuncts.
"""

import regex as re
import unicodedata
import itertools


def graphemes(text: str) -> list[str]:
    """
    Extract grapheme clusters from text using Unicode regex.
    
    Uses the \\X pattern which matches extended grapheme clusters,
    combined with NFC normalization to ensure proper handling of
    Hindi combining characters.
    
    Args:
        text: Input text string
        
    Returns:
        List of grapheme clusters
    """
    return re.findall(r'\X', unicodedata.normalize("NFC", text))


def build_hindi_grapheme_set() -> set[str]:
    """
    Build a comprehensive set of possible Hindi grapheme combinations.
    
    This generates all possible combinations of Hindi linguistic components:
    - Independent vowels
    - Bare consonants
    - Consonant + matra combinations
    - Consonant conjuncts (C + virama + C)
    - With nasals and visarga
    
    Returns:
        Set of all possible grapheme strings
        
    Note:
        This generates ~50k+ possible graphemes. For practical use,
        it's better to extract unique graphemes from actual training data.
    """
    vowels = list("अआइईउऊएऐओऔऋॠऌॡ")
    consonants = list("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
    matras = list("ािीुूेैोौृॄॅॉॢॣ")
    extras = list("ंःँ")
    virama = "्"
    
    graphemes_set = set()
    
    # Independent vowels
    graphemes_set.update(vowels)
    
    # Bare consonants
    graphemes_set.update(consonants)
    
    # Consonant + matra / nasal / visarga
    for c in consonants:
        for m in matras:
            graphemes_set.add(c + m)
            for e in extras:
                graphemes_set.add(c + m + e)
        for e in extras:
            graphemes_set.add(c + e)
    
    # Consonant conjuncts (C + Virama + C)
    for c1, c2 in itertools.product(consonants, repeat=2):
        graphemes_set.add(c1 + virama + c2)
        for m in matras:
            graphemes_set.add(c1 + virama + c2 + m)
            for e in extras:
                graphemes_set.add(c1 + virama + c2 + m + e)
    
    return graphemes_set
