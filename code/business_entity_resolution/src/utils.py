"""
Utility functions for text preprocessing and normalization.
"""

import re
import unicodedata
from typing import List, Set


def normalize_text(text: str) -> str:
    """
    Normalize text: lowercase, remove extra spaces, remove special chars.
    Preserves word boundaries for token-based matching.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Handle common abbreviations
    abbrev_map = {
        r'\bcorp\b': 'corporation',
        r'\bltd\b': 'limited',
        r'\bpvt\b': 'private',
        r'\binc\b': 'incorporated',
        r'\bllc\b': 'limited liability company',
        r'\bco\b': 'company',
        r'\b&\b': 'and',
        r'\bst\b': 'street',
        r'\brd\b': 'road',
        r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard',
        r'\bdr\b': 'drive',
        r'\bln\b': 'lane',
        r'\bpk\b': 'park',
        r'\bplz\b': 'plaza',
    }
    
    for abbrev, full in abbrev_map.items():
        text = re.sub(abbrev, full, text)
    
    # Remove accents/diacritics
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    
    # Remove punctuation except spaces (spaces are word separators)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize(text: str) -> Set[str]:
    """Split normalized text into tokens (words)."""
    return set(normalize_text(text).split())


def get_tokens_list(text: str) -> List[str]:
    """Split normalized text into ordered token list."""
    return normalize_text(text).split()


def soundex(text: str) -> str:
    """
    Encode text using Soundex algorithm for phonetic matching.
    Useful for name variations (Smith, Smyth, etc.)
    """
    text = normalize_text(text).upper()
    if not text:
        return ""
    
    # Keep first letter
    first = text[0]
    
    # Mapping
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    
    # Encode rest
    code = first
    prev = mapping.get(first, '0')
    
    for char in text[1:]:
        digit = mapping.get(char, '0')
        if digit != '0' and digit != prev:
            code += digit
        if digit != '0':
            prev = digit
    
    # Pad with zeros or truncate to 4 chars
    code = (code + '000')[:4]
    
    return code


def metaphone(text: str) -> str:
    """
    Simple Metaphone implementation for phonetic matching.
    More sophisticated than Soundex.
    """
    text = normalize_text(text).upper()
    if not text:
        return ""
    
    # Drop duplicate consecutive letters
    text = re.sub(r'(.)\1+', r'\1', text)
    
    # Simple replacement rules
    text = text.replace('DG', 'G')
    text = text.replace('GH', '')
    text = text.replace('GN', 'N')
    text = text.replace('KN', 'N')
    text = text.replace('PH', 'F')
    text = text.replace('PS', 'S')
    text = text.replace('TCH', 'CH')
    
    # Drop vowels except at start
    if len(text) > 0:
        text = text[0] + re.sub(r'[AEIOUWY]', '', text[1:])
    
    # Keep first 4
    return (text + '0000')[:4]


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer than s2
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Compute Levenshtein similarity ratio [0, 1]."""
    s1 = normalize_text(s1)
    s2 = normalize_text(s2)
    
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    
    return 1.0 - (distance / max_len)


def token_overlap_ratio(text1: str, text2: str) -> float:
    """Compute ratio of overlapping tokens."""
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    overlap = len(tokens1 & tokens2)
    total = len(tokens1 | tokens2)
    
    return overlap / total if total > 0 else 0.0
