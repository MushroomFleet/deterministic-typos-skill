#!/usr/bin/env python3
"""
apply_typos.py - Deterministic typo application using zerobytes methodology

Usage:
    python apply_typos.py input.txt --seed 42 --intensity moderate --profile balanced
    python apply_typos.py input.txt --seed 42 --output output.txt
    
    # Dual seed mode
    python apply_typos.py input.txt --position-seed 100 --type-seed 200
"""

import re
import argparse
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from pathlib import Path

from typo_hash import (
    document_seed,
    should_typo_at,
    select_typo_type,
    select_adjacent_key,
    select_sticky_behavior,
    select_variant,
    fatigue_intensity,
    hash_to_float,
    position_hash,
    VARIANT_SALT
)

# =============================================================================
# QWERTY ADJACENCY MAP
# =============================================================================

QWERTY_ADJACENT: Dict[str, List[str]] = {
    'q': ['w', 'a'],
    'w': ['q', 'e', 'a', 's'],
    'e': ['w', 'r', 's', 'd'],
    'r': ['e', 't', 'd', 'f'],
    't': ['r', 'y', 'f', 'g'],
    'y': ['t', 'u', 'g', 'h'],
    'u': ['y', 'i', 'h', 'j'],
    'i': ['u', 'o', 'j', 'k'],
    'o': ['i', 'p', 'k', 'l'],
    'p': ['o', 'l'],
    'a': ['q', 'w', 's', 'z'],
    's': ['a', 'w', 'e', 'd', 'z', 'x'],
    'd': ['s', 'e', 'r', 'f', 'x', 'c'],
    'f': ['d', 'r', 't', 'g', 'c', 'v'],
    'g': ['f', 't', 'y', 'h', 'v', 'b'],
    'h': ['g', 'y', 'u', 'j', 'b', 'n'],
    'j': ['h', 'u', 'i', 'k', 'n', 'm'],
    'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p'],
    'z': ['a', 's', 'x'],
    'x': ['z', 's', 'd', 'c'],
    'c': ['x', 'd', 'f', 'v'],
    'v': ['c', 'f', 'g', 'b'],
    'b': ['v', 'g', 'h', 'n'],
    'n': ['b', 'h', 'j', 'm'],
    'm': ['n', 'j', 'k'],
}

# High-frequency letters for sticky keys (weighted)
STICKY_WEIGHTS: Dict[str, float] = {
    'e': 0.15, 't': 0.12, 'a': 0.10, 'o': 0.09, 'i': 0.09,
    'n': 0.08, 's': 0.07, 'r': 0.07, 'h': 0.06, 'l': 0.05,
    'd': 0.04, 'c': 0.04, 'u': 0.04, 'm': 0.03, 'w': 0.03,
    'f': 0.03, 'g': 0.02, 'y': 0.02, 'p': 0.02, 'b': 0.02,
}

# Common transposition pairs
TRANSPOSITION_PAIRS: List[Tuple[str, str]] = [
    ('t', 'h'), ('h', 'e'), ('i', 'e'), ('e', 'r'), ('e', 's'),
    ('e', 'd'), ('i', 'n'), ('o', 'n'), ('a', 'n'), ('n', 'g'),
]

# =============================================================================
# PROFILE DEFINITIONS
# =============================================================================

PROFILES: Dict[str, Dict[str, float]] = {
    'balanced': {
        'sticky_key': 0.30,
        'adjacent_key': 0.35,
        'transposition': 0.25,
        'dropped_letter': 0.10,
    },
    'keyboard-hardware': {
        'sticky_key': 0.60,
        'adjacent_key': 0.25,
        'transposition': 0.10,
        'dropped_letter': 0.05,
    },
    'speed-typist': {
        'sticky_key': 0.05,
        'adjacent_key': 0.50,
        'transposition': 0.20,
        'dropped_letter': 0.25,
    },
    'fatigue': {
        'sticky_key': 0.25,
        'adjacent_key': 0.30,
        'transposition': 0.25,
        'dropped_letter': 0.20,
    },
}

INTENSITY_VALUES: Dict[str, float] = {
    'light': 0.015,      # 1.5% of characters
    'moderate': 0.04,    # 4% of characters
    'heavy': 0.085,      # 8.5% of characters
}

# =============================================================================
# PRESERVATION PATTERNS
# =============================================================================

PRESERVE_PATTERNS = [
    r'https?://\S+',                    # URLs
    r'\S+@\S+\.\S+',                    # Emails
    r'```[\s\S]*?```',                  # Fenced code blocks
    r'`[^`]+`',                         # Inline code
    r'\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b',  # Numbers
    r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}',  # Dates
]

# Markdown syntax characters to preserve in context
MARKDOWN_CHARS = set('#*_[]()>`-')


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TypoConfig:
    """Configuration for typo application."""
    intensity: float = 0.04
    profile: str = 'balanced'
    seed: Optional[int] = None
    position_seed: Optional[int] = None
    type_seed: Optional[int] = None
    dyslexic: bool = False
    punctuation: bool = False


@dataclass 
class CharInfo:
    """Information about a character position."""
    index: int
    char: str
    preserved: bool
    word_start: bool
    word_end: bool


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def find_preserved_ranges(text: str) -> Set[int]:
    """
    Find character indices that should not be modified.
    
    Returns:
        Set of indices to preserve
    """
    preserved = set()
    
    for pattern in PRESERVE_PATTERNS:
        for match in re.finditer(pattern, text):
            for i in range(match.start(), match.end()):
                preserved.add(i)
    
    return preserved


def analyze_text(text: str) -> List[CharInfo]:
    """
    Analyze text and create CharInfo for each position.
    
    Args:
        text: Input text
    
    Returns:
        List of CharInfo objects
    """
    preserved = find_preserved_ranges(text)
    chars = []
    
    for i, char in enumerate(text):
        # Check if at word boundary
        prev_char = text[i-1] if i > 0 else ' '
        next_char = text[i+1] if i < len(text)-1 else ' '
        
        word_start = not prev_char.isalpha() and char.isalpha()
        word_end = char.isalpha() and not next_char.isalpha()
        
        chars.append(CharInfo(
            index=i,
            char=char,
            preserved=i in preserved,
            word_start=word_start,
            word_end=word_end
        ))
    
    return chars


def apply_sticky_key(
    text: str, 
    pos: int, 
    seed: int,
    type_seed: Optional[int] = None
) -> str:
    """Apply sticky key typo (double or drop letter)."""
    char = text[pos]
    behavior = select_sticky_behavior(pos, seed, type_seed=type_seed)
    
    if behavior == 'double':
        return text[:pos] + char + text[pos:]
    else:  # drop
        return text[:pos] + text[pos+1:]


def apply_adjacent_key(
    text: str,
    pos: int,
    seed: int,
    type_seed: Optional[int] = None
) -> str:
    """Apply adjacent key typo (substitute or insert)."""
    char = text[pos]
    adjacent = select_adjacent_key(pos, char, seed, QWERTY_ADJACENT, type_seed)
    
    # Decide: substitute or insert
    h = position_hash(pos, type_seed or seed, VARIANT_SALT + 1)
    if hash_to_float(h) < 0.7:
        # Substitute
        return text[:pos] + adjacent + text[pos+1:]
    else:
        # Insert after
        return text[:pos+1] + adjacent + text[pos+1:]


def apply_transposition(text: str, pos: int) -> str:
    """Swap character at pos with character at pos+1."""
    if pos >= len(text) - 1:
        return text
    return text[:pos] + text[pos+1] + text[pos] + text[pos+2:]


def apply_dropped_letter(text: str, pos: int) -> str:
    """Remove character at position."""
    return text[:pos] + text[pos+1:]


def apply_typo(
    text: str,
    pos: int,
    typo_type: str,
    seed: int,
    type_seed: Optional[int] = None
) -> Tuple[str, int]:
    """
    Apply a single typo at the specified position.
    
    Returns:
        Tuple of (modified_text, length_delta)
    """
    original_len = len(text)
    
    if typo_type == 'sticky_key':
        text = apply_sticky_key(text, pos, seed, type_seed)
    elif typo_type == 'adjacent_key':
        text = apply_adjacent_key(text, pos, seed, type_seed)
    elif typo_type == 'transposition':
        text = apply_transposition(text, pos)
    elif typo_type == 'dropped_letter':
        text = apply_dropped_letter(text, pos)
    
    return text, len(text) - original_len


def apply_typos(
    text: str,
    config: TypoConfig
) -> Tuple[str, int]:
    """
    Apply typos to text using deterministic zerobytes methodology.
    
    Args:
        text: Input text
        config: Typo configuration
    
    Returns:
        Tuple of (modified_text, seed_used)
    """
    # Determine seed
    seed = config.seed if config.seed is not None else document_seed(text)
    position_seed = config.position_seed if config.position_seed is not None else seed
    type_seed = config.type_seed if config.type_seed is not None else seed
    
    # Get profile weights
    profile_weights = PROFILES.get(config.profile, PROFILES['balanced'])
    
    # Analyze text
    char_infos = analyze_text(text)
    
    # Collect typo positions and types (before applying, to maintain determinism)
    typos_to_apply: List[Tuple[int, str]] = []
    
    for info in char_infos:
        # Skip preserved characters
        if info.preserved:
            continue
        
        # Skip non-alphabetic for most typos
        if not info.char.isalpha():
            continue
        
        # Calculate effective intensity (for fatigue profile)
        if config.profile == 'fatigue':
            effective_intensity = fatigue_intensity(
                info.index, len(text), seed, config.intensity
            )
        else:
            effective_intensity = config.intensity
        
        # Determine if typo occurs here
        if should_typo_at(info.index, seed, effective_intensity, position_seed):
            typo_type = select_typo_type(info.index, seed, profile_weights, type_seed)
            
            # Validate typo type for context
            if typo_type == 'transposition' and info.word_end:
                continue  # Don't transpose at word end
            if typo_type == 'dropped_letter' and (info.word_start or info.word_end):
                continue  # Don't drop first/last letters
            if typo_type == 'sticky_key' and info.char.lower() not in STICKY_WEIGHTS:
                continue  # Only sticky on weighted letters
            
            typos_to_apply.append((info.index, typo_type))
    
    # Apply typos in reverse order (to preserve indices)
    result = text
    offset = 0
    
    # Sort by position, then reverse
    typos_to_apply.sort(key=lambda x: x[0], reverse=True)
    
    for pos, typo_type in typos_to_apply:
        result, delta = apply_typo(result, pos, typo_type, seed, type_seed)
    
    return result, seed


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_intensity(value: str) -> float:
    """Parse intensity from string (name or percentage)."""
    if value in INTENSITY_VALUES:
        return INTENSITY_VALUES[value]
    try:
        # Try parsing as percentage
        if value.endswith('%'):
            return float(value[:-1]) / 100
        return float(value)
    except ValueError:
        return INTENSITY_VALUES['moderate']


def main():
    parser = argparse.ArgumentParser(
        description='Apply deterministic typos to text using zerobytes methodology'
    )
    parser.add_argument('input', help='Input file path')
    parser.add_argument('--output', '-o', help='Output file path (default: stdout)')
    parser.add_argument('--seed', '-s', type=int, help='World seed for determinism')
    parser.add_argument('--position-seed', type=int, help='Override seed for typo locations')
    parser.add_argument('--type-seed', type=int, help='Override seed for typo types')
    parser.add_argument(
        '--intensity', '-i', 
        default='moderate',
        help='Typo density: light, moderate, heavy, or percentage (default: moderate)'
    )
    parser.add_argument(
        '--profile', '-p',
        default='balanced',
        choices=['balanced', 'keyboard-hardware', 'speed-typist', 'fatigue'],
        help='Typo pattern profile (default: balanced)'
    )
    parser.add_argument('--dyslexic', action='store_true', help='Enable dyslexic patterns')
    parser.add_argument('--punctuation', action='store_true', help='Enable punctuation habits')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress seed output')
    
    args = parser.parse_args()
    
    # Read input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    text = input_path.read_text(encoding='utf-8')
    
    # Create config
    config = TypoConfig(
        intensity=parse_intensity(args.intensity),
        profile=args.profile,
        seed=args.seed,
        position_seed=args.position_seed,
        type_seed=args.type_seed,
        dyslexic=args.dyslexic,
        punctuation=args.punctuation
    )
    
    # Apply typos
    result, seed_used = apply_typos(text, config)
    
    # Output
    if args.output:
        Path(args.output).write_text(result, encoding='utf-8')
        if not args.quiet:
            print(f"Output written to: {args.output}")
            print(f"Seed used: {seed_used}")
    else:
        print(result)
        if not args.quiet:
            print(f"\n---\nSeed: {seed_used}", file=__import__('sys').stderr)
    
    return 0


if __name__ == '__main__':
    exit(main())
