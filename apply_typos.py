#!/usr/bin/env python3
"""
apply_typos.py - Deterministic typo application using zerobytes methodology

CRITICAL: This script MUST be executed directly. Claude should NOT attempt
to "implement" this algorithm mentally - that breaks determinism.

Usage:
    python apply_typos.py input.txt --seed 42
    python apply_typos.py input.txt --seed 42 --output output.txt
    python apply_typos.py input.txt --seed 42 --intensity moderate --profile balanced
"""

import re
import argparse
import struct
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from pathlib import Path

try:
    import xxhash
    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False
    import hashlib

# =============================================================================
# SALT CONSTANTS - Each decision domain gets unique salt
# =============================================================================

LOCATION_SALT = 0x1000
TYPE_SALT = 0x2000
VARIANT_SALT = 0x3000
STICKY_SALT = 0x4000
ADJACENT_SALT = 0x5000
SUBSTITUTE_SALT = 0x6000

# =============================================================================
# CORE HASH FUNCTIONS (inlined for single-file operation)
# =============================================================================

def position_hash(position: int, seed: int, salt: int = 0) -> int:
    """Deterministic hash from position and seed."""
    combined_seed = (seed ^ salt) & 0xFFFFFFFF
    
    if HAS_XXHASH:
        h = xxhash.xxh32(seed=combined_seed)
        h.update(struct.pack('<I', position & 0xFFFFFFFF))
        return h.intdigest()
    else:
        data = struct.pack('<II', combined_seed, position & 0xFFFFFFFF)
        return int(hashlib.md5(data).hexdigest()[:8], 16)


def hash_to_float(h: int) -> float:
    """Convert hash to float in [0.0, 1.0)."""
    return (h & 0xFFFFFFFF) / 0x100000000


def document_seed(content: str) -> int:
    """Generate deterministic seed from document content."""
    if HAS_XXHASH:
        return xxhash.xxh32(content.encode('utf-8')).intdigest()
    else:
        return int(hashlib.md5(content.encode('utf-8')).hexdigest()[:8], 16)


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

# Letters eligible for sticky key errors
STICKY_LETTERS = set('etaoinsrhldcumwfgypb')

# =============================================================================
# PROFILE DEFINITIONS
# =============================================================================

PROFILES: Dict[str, Dict[str, float]] = {
    'balanced': {
        'sticky_double': 0.18,
        'sticky_drop': 0.12,
        'adjacent_sub': 0.25,
        'adjacent_ins': 0.10,
        'transposition': 0.25,
        'dropped_letter': 0.10,
    },
    'keyboard-hardware': {
        'sticky_double': 0.42,
        'sticky_drop': 0.18,
        'adjacent_sub': 0.18,
        'adjacent_ins': 0.07,
        'transposition': 0.10,
        'dropped_letter': 0.05,
    },
    'speed-typist': {
        'sticky_double': 0.03,
        'sticky_drop': 0.02,
        'adjacent_sub': 0.35,
        'adjacent_ins': 0.15,
        'transposition': 0.20,
        'dropped_letter': 0.25,
    },
    'fatigue': {
        'sticky_double': 0.15,
        'sticky_drop': 0.10,
        'adjacent_sub': 0.22,
        'adjacent_ins': 0.08,
        'transposition': 0.25,
        'dropped_letter': 0.20,
    },
}

INTENSITY_VALUES: Dict[str, float] = {
    'light': 0.015,
    'moderate': 0.04,
    'heavy': 0.085,
}

# =============================================================================
# PRESERVATION PATTERNS
# =============================================================================

PRESERVE_PATTERNS = [
    r'https?://\S+',
    r'\S+@\S+\.\S+',
    r'```[\s\S]*?```',
    r'`[^`]+`',
    r'\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b',
    r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}',
    r'"[^"]*"',  # Preserve quoted strings
]


# =============================================================================
# DETERMINISTIC TYPO APPLICATION
# =============================================================================

def find_preserved_indices(text: str) -> Set[int]:
    """Find all character indices that should not be modified."""
    preserved = set()
    for pattern in PRESERVE_PATTERNS:
        for match in re.finditer(pattern, text):
            for i in range(match.start(), match.end()):
                preserved.add(i)
    return preserved


def get_typo_decision(char_idx: int, seed: int, intensity: float, profile_weights: Dict[str, float]) -> Optional[str]:
    """
    Pure deterministic decision: should this position have a typo, and what type?
    
    Returns typo type string or None if no typo at this position.
    ALL decisions derived purely from (char_idx, seed) - no external state.
    """
    # Decision 1: Should there be a typo here at all?
    location_hash = position_hash(char_idx, seed, LOCATION_SALT)
    if hash_to_float(location_hash) >= intensity:
        return None
    
    # Decision 2: What type of typo?
    type_hash = position_hash(char_idx, seed, TYPE_SALT)
    roll = hash_to_float(type_hash)
    
    cumulative = 0.0
    for typo_type, weight in profile_weights.items():
        cumulative += weight
        if roll < cumulative:
            return typo_type
    
    return list(profile_weights.keys())[-1]


def get_adjacent_key(char: str, char_idx: int, seed: int) -> str:
    """Deterministically select an adjacent key."""
    char_lower = char.lower()
    if char_lower not in QWERTY_ADJACENT:
        return char
    
    adjacent = QWERTY_ADJACENT[char_lower]
    adj_hash = position_hash(char_idx, seed, ADJACENT_SALT)
    selected = adjacent[adj_hash % len(adjacent)]
    
    return selected.upper() if char.isupper() else selected


def apply_typos_deterministic(text: str, seed: int, intensity: float, profile: str) -> str:
    """
    Apply typos with perfect determinism.
    
    Key insight: We must make ALL decisions in a first pass based purely on
    original character positions, then apply modifications in reverse order.
    """
    profile_weights = PROFILES.get(profile, PROFILES['balanced'])
    preserved = find_preserved_indices(text)
    
    # First pass: Collect ALL typo decisions based on original positions
    # Each decision is a tuple: (original_position, typo_type, extra_data)
    typo_plan: List[Tuple[int, str, any]] = []
    
    for i, char in enumerate(text):
        # Skip if preserved
        if i in preserved:
            continue
        
        # Skip non-alphabetic characters
        if not char.isalpha():
            continue
        
        # Get deterministic typo decision for this position
        typo_type = get_typo_decision(i, seed, intensity, profile_weights)
        
        if typo_type is None:
            continue
        
        # Validate typo type is applicable to this character
        # These checks are deterministic based on the character at position i
        char_lower = char.lower()
        
        if typo_type in ('sticky_double', 'sticky_drop'):
            if char_lower not in STICKY_LETTERS:
                continue
        
        if typo_type in ('adjacent_sub', 'adjacent_ins'):
            if char_lower not in QWERTY_ADJACENT:
                continue
        
        if typo_type == 'transposition':
            # Need next char to exist and be alphabetic
            if i >= len(text) - 1 or not text[i + 1].isalpha():
                continue
            # Don't transpose if next position is preserved
            if (i + 1) in preserved:
                continue
        
        if typo_type == 'dropped_letter':
            # Check we're not at word boundary (deterministic from text)
            prev_alpha = (i > 0 and text[i-1].isalpha())
            next_alpha = (i < len(text) - 1 and text[i+1].isalpha())
            if not (prev_alpha and next_alpha):
                continue
        
        # Compute any extra data needed for this typo type
        extra = None
        if typo_type in ('adjacent_sub', 'adjacent_ins'):
            extra = get_adjacent_key(char, i, seed)
        
        typo_plan.append((i, typo_type, extra))
    
    # Second pass: Apply typos in REVERSE order to preserve indices
    result = list(text)
    
    for pos, typo_type, extra in reversed(typo_plan):
        if pos >= len(result):
            continue
            
        char = result[pos]
        
        if typo_type == 'sticky_double':
            # Double the letter
            result.insert(pos + 1, char)
        
        elif typo_type == 'sticky_drop':
            # Remove the letter
            result.pop(pos)
        
        elif typo_type == 'adjacent_sub':
            # Substitute with adjacent key
            result[pos] = extra
        
        elif typo_type == 'adjacent_ins':
            # Insert adjacent key after
            result.insert(pos + 1, extra)
        
        elif typo_type == 'transposition':
            # Swap with next character
            if pos + 1 < len(result):
                result[pos], result[pos + 1] = result[pos + 1], result[pos]
        
        elif typo_type == 'dropped_letter':
            # Remove the letter
            result.pop(pos)
    
    return ''.join(result)


def apply_typos(text: str, seed: int, intensity: float = 0.04, profile: str = 'balanced') -> str:
    """Main entry point for applying typos."""
    return apply_typos_deterministic(text, seed, intensity, profile)


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_determinism(text: str, seed: int, intensity: float = 0.04, runs: int = 10) -> bool:
    """Verify that multiple runs produce identical output."""
    results = set()
    for _ in range(runs):
        output = apply_typos(text, seed, intensity)
        results.add(output)
    
    if len(results) == 1:
        print(f"✓ DETERMINISTIC: {runs} runs produced identical output")
        return True
    else:
        print(f"✗ NON-DETERMINISTIC: {runs} runs produced {len(results)} different outputs")
        return False


# =============================================================================
# CLI
# =============================================================================

def parse_intensity(value: str) -> float:
    """Parse intensity from string."""
    if value in INTENSITY_VALUES:
        return INTENSITY_VALUES[value]
    try:
        if value.endswith('%'):
            return float(value[:-1]) / 100
        return float(value)
    except ValueError:
        return INTENSITY_VALUES['moderate']


def main():
    parser = argparse.ArgumentParser(
        description='Apply deterministic typos using zerobytes methodology'
    )
    parser.add_argument('input', help='Input file path')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--seed', '-s', type=int, help='Seed for determinism')
    parser.add_argument('--intensity', '-i', default='moderate',
                        help='light, moderate, heavy, or percentage')
    parser.add_argument('--profile', '-p', default='balanced',
                        choices=['balanced', 'keyboard-hardware', 'speed-typist', 'fatigue'])
    parser.add_argument('--verify', action='store_true', 
                        help='Run determinism verification')
    parser.add_argument('--quiet', '-q', action='store_true')
    
    args = parser.parse_args()
    
    # Read input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {args.input}")
        return 1
    
    text = input_path.read_text(encoding='utf-8')
    
    # Determine seed
    seed = args.seed if args.seed is not None else document_seed(text)
    intensity = parse_intensity(args.intensity)
    
    # Verify mode
    if args.verify:
        verify_determinism(text, seed, intensity)
        return 0
    
    # Apply typos
    result = apply_typos(text, seed, intensity, args.profile)
    
    # Output
    if args.output:
        Path(args.output).write_text(result, encoding='utf-8')
        if not args.quiet:
            print(f"Seed: {seed}")
            print(f"Output: {args.output}")
    else:
        print(result)
        if not args.quiet:
            import sys
            print(f"\n---\nSeed: {seed}", file=sys.stderr)
    
    return 0


if __name__ == '__main__':
    exit(main())
