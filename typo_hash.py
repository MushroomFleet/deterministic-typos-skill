#!/usr/bin/env python3
"""
typo_hash.py - Zerobytes position-as-seed typo generation

Core principle: character position IS the seed coordinate.
Same document + same seed = identical typos every time.

Uses xxhash32 for fast, deterministic, cross-platform hashing.
"""

import struct
from typing import List, Dict, Tuple, Optional

try:
    import xxhash
    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False
    import hashlib

# =============================================================================
# SALT CONSTANTS - Each decision domain gets unique salt
# =============================================================================

LOCATION_SALT = 0x1000    # WHERE typos occur
TYPE_SALT = 0x2000        # WHAT pattern applies
VARIANT_SALT = 0x3000     # WHICH specific variant
INTENSITY_SALT = 0x4000   # Fatigue curve modulation
ADJACENT_SALT = 0x5000    # Adjacent key selection
STICKY_SALT = 0x6000      # Sticky key behavior (double/drop)

# =============================================================================
# CORE HASH FUNCTIONS
# =============================================================================

def position_hash(position: int, seed: int, salt: int = 0) -> int:
    """
    Generate deterministic hash from position and seed.
    
    Args:
        position: Character index in document (0-indexed)
        seed: World seed (user-provided or document-derived)
        salt: Domain-specific salt to separate decision spaces
    
    Returns:
        32-bit unsigned integer hash
    """
    combined_seed = (seed ^ salt) & 0xFFFFFFFF
    
    if HAS_XXHASH:
        h = xxhash.xxh32(seed=combined_seed)
        h.update(struct.pack('<I', position & 0xFFFFFFFF))
        return h.intdigest()
    else:
        # Fallback: deterministic but slower
        data = struct.pack('<II', combined_seed, position & 0xFFFFFFFF)
        return int(hashlib.md5(data).hexdigest()[:8], 16)


def position_hash_2d(x: int, y: int, seed: int, salt: int = 0) -> int:
    """
    2D position hash for paragraph/word hierarchies.
    
    Args:
        x: First coordinate (e.g., paragraph index)
        y: Second coordinate (e.g., word index within paragraph)
        seed: World seed
        salt: Domain-specific salt
    
    Returns:
        32-bit unsigned integer hash
    """
    combined_seed = (seed ^ salt) & 0xFFFFFFFF
    
    if HAS_XXHASH:
        h = xxhash.xxh32(seed=combined_seed)
        h.update(struct.pack('<II', x & 0xFFFFFFFF, y & 0xFFFFFFFF))
        return h.intdigest()
    else:
        data = struct.pack('<III', combined_seed, x & 0xFFFFFFFF, y & 0xFFFFFFFF)
        return int(hashlib.md5(data).hexdigest()[:8], 16)


def hash_to_float(h: int) -> float:
    """
    Convert hash to float in range [0.0, 1.0).
    
    Args:
        h: Integer hash value
    
    Returns:
        Float between 0.0 (inclusive) and 1.0 (exclusive)
    """
    return (h & 0xFFFFFFFF) / 0x100000000


def document_seed(content: str) -> int:
    """
    Generate deterministic seed from document content.
    
    Used when user doesn't provide explicit seed.
    
    Args:
        content: Full document text
    
    Returns:
        32-bit seed derived from content
    """
    if HAS_XXHASH:
        return xxhash.xxh32(content.encode('utf-8')).intdigest()
    else:
        return int(hashlib.md5(content.encode('utf-8')).hexdigest()[:8], 16)


# =============================================================================
# TYPO DECISION FUNCTIONS
# =============================================================================

def should_typo_at(
    char_index: int, 
    seed: int, 
    intensity: float,
    position_seed: Optional[int] = None
) -> bool:
    """
    Deterministically decide if a typo occurs at this character position.
    
    Args:
        char_index: Position in document (0-indexed)
        seed: World seed
        intensity: Typo density (0.0 to 1.0, e.g., 0.05 for 5%)
        position_seed: Override seed for location decisions only
    
    Returns:
        True if typo should occur at this position
    """
    effective_seed = position_seed if position_seed is not None else seed
    h = position_hash(char_index, effective_seed, LOCATION_SALT)
    threshold = hash_to_float(h)
    return threshold < intensity


def select_typo_type(
    char_index: int,
    seed: int,
    profile_weights: Dict[str, float],
    type_seed: Optional[int] = None
) -> str:
    """
    Deterministically select which typo pattern to apply.
    
    Args:
        char_index: Position in document
        seed: World seed
        profile_weights: Dict mapping type name -> weight (must sum to 1.0)
        type_seed: Override seed for type decisions only
    
    Returns:
        Selected typo type string
    """
    effective_seed = type_seed if type_seed is not None else seed
    h = position_hash(char_index, effective_seed, TYPE_SALT)
    roll = hash_to_float(h)
    
    cumulative = 0.0
    for typo_type, weight in profile_weights.items():
        cumulative += weight
        if roll < cumulative:
            return typo_type
    
    # Fallback to last type
    return list(profile_weights.keys())[-1]


def select_variant(
    char_index: int,
    seed: int,
    options: List[str],
    type_seed: Optional[int] = None
) -> str:
    """
    Deterministically select from a list of variant options.
    
    Args:
        char_index: Position in document
        seed: World seed
        options: List of possible variants
        type_seed: Override seed for variant decisions
    
    Returns:
        Selected variant from options list
    """
    if not options:
        return ""
    
    effective_seed = type_seed if type_seed is not None else seed
    h = position_hash(char_index, effective_seed, VARIANT_SALT)
    index = h % len(options)
    return options[index]


def select_adjacent_key(
    char_index: int,
    original_char: str,
    seed: int,
    adjacency_map: Dict[str, List[str]],
    type_seed: Optional[int] = None
) -> str:
    """
    Deterministically select adjacent key for substitution/insertion.
    
    Args:
        char_index: Position in document
        original_char: The character being modified
        seed: World seed
        adjacency_map: Dict mapping char -> list of adjacent chars
        type_seed: Override seed
    
    Returns:
        Adjacent character to use
    """
    char_lower = original_char.lower()
    if char_lower not in adjacency_map:
        return original_char
    
    adjacent_keys = adjacency_map[char_lower]
    if not adjacent_keys:
        return original_char
    
    effective_seed = type_seed if type_seed is not None else seed
    
    # Include original char in hash for extra variation
    if HAS_XXHASH:
        h = xxhash.xxh32(seed=(effective_seed ^ ADJACENT_SALT) & 0xFFFFFFFF)
        h.update(struct.pack('<I', char_index & 0xFFFFFFFF))
        h.update(original_char.encode('utf-8'))
        hash_val = h.intdigest()
    else:
        data = struct.pack('<II', (effective_seed ^ ADJACENT_SALT) & 0xFFFFFFFF, char_index)
        data += original_char.encode('utf-8')
        hash_val = int(hashlib.md5(data).hexdigest()[:8], 16)
    
    index = hash_val % len(adjacent_keys)
    result = adjacent_keys[index]
    
    # Preserve case
    if original_char.isupper():
        result = result.upper()
    
    return result


def select_sticky_behavior(
    char_index: int,
    seed: int,
    double_probability: float = 0.6,
    type_seed: Optional[int] = None
) -> str:
    """
    Deterministically decide sticky key behavior: double or drop.
    
    Args:
        char_index: Position in document
        seed: World seed
        double_probability: Probability of doubling vs dropping (default 60%)
        type_seed: Override seed
    
    Returns:
        'double' or 'drop'
    """
    effective_seed = type_seed if type_seed is not None else seed
    h = position_hash(char_index, effective_seed, STICKY_SALT)
    roll = hash_to_float(h)
    return 'double' if roll < double_probability else 'drop'


# =============================================================================
# FATIGUE PROFILE SUPPORT
# =============================================================================

def fatigue_intensity(
    char_index: int,
    doc_length: int,
    seed: int,
    base_intensity: float,
    start_multiplier: float = 0.3,
    end_multiplier: float = 1.5
) -> float:
    """
    Calculate position-aware intensity for fatigue profile.
    
    Uses coherent noise for natural variation over smooth curve.
    
    Args:
        char_index: Position in document
        doc_length: Total document length
        seed: World seed
        base_intensity: Base typo density
        start_multiplier: Intensity multiplier at document start
        end_multiplier: Intensity multiplier at document end
    
    Returns:
        Adjusted intensity for this position (0.0 to 1.0)
    """
    if doc_length == 0:
        return base_intensity
    
    # Normalize position to 0.0 - 1.0
    normalized_pos = char_index / doc_length
    
    # Exponential fatigue curve
    curve = start_multiplier + (normalized_pos ** 2) * (end_multiplier - start_multiplier)
    base_fatigue = base_intensity * curve
    
    # Add coherent noise for natural variation
    noise = coherent_noise_1d(char_index * 0.01, seed)
    variation = noise * 0.3 * base_fatigue
    
    return max(0.0, min(1.0, base_fatigue + variation))


def coherent_noise_1d(x: float, seed: int, octaves: int = 3) -> float:
    """
    1D coherent noise for smooth intensity variation.
    
    Args:
        x: Position (typically char_index * small_scale)
        seed: World seed
        octaves: Number of noise layers
    
    Returns:
        Noise value in range [-1.0, 1.0]
    """
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0
    
    for i in range(octaves):
        x0 = int(x * frequency)
        
        # Smoothstep interpolation factor
        sx = (x * frequency) % 1
        sx = sx * sx * (3 - 2 * sx)
        
        # Hash at integer positions
        n0 = hash_to_float(position_hash(x0, seed + i, INTENSITY_SALT)) * 2 - 1
        n1 = hash_to_float(position_hash(x0 + 1, seed + i, INTENSITY_SALT)) * 2 - 1
        
        # Interpolate
        value += amplitude * (n0 * (1 - sx) + n1 * sx)
        max_amplitude += amplitude
        
        amplitude *= 0.5
        frequency *= 2.0
    
    return value / max_amplitude if max_amplitude > 0 else 0.0


# =============================================================================
# VERIFICATION UTILITIES
# =============================================================================

def verify_determinism(
    apply_fn,
    text: str,
    seed: int,
    iterations: int = 10
) -> Tuple[bool, str]:
    """
    Verify that typo application is deterministic.
    
    Args:
        apply_fn: Function that takes (text, seed) and returns modified text
        text: Input text to test
        seed: Seed to test with
        iterations: Number of times to run
    
    Returns:
        Tuple of (is_deterministic: bool, message: str)
    """
    results = set()
    
    for _ in range(iterations):
        output = apply_fn(text, seed)
        results.add(output)
    
    if len(results) == 1:
        return True, f"✓ Deterministic: {iterations} runs produced identical output"
    else:
        return False, f"✗ Non-deterministic: {iterations} runs produced {len(results)} unique outputs"


def verify_order_independence(
    should_typo_fn,
    positions: List[int],
    seed: int,
    intensity: float
) -> Tuple[bool, str]:
    """
    Verify that position processing order doesn't affect results.
    
    Args:
        should_typo_fn: Function to test (typically should_typo_at)
        positions: List of positions to check
        seed: Seed to test with
        intensity: Intensity to test with
    
    Returns:
        Tuple of (is_order_independent: bool, message: str)
    """
    # Forward order
    forward = {pos: should_typo_fn(pos, seed, intensity) for pos in positions}
    
    # Reverse order
    reverse = {pos: should_typo_fn(pos, seed, intensity) for pos in reversed(positions)}
    
    # Random-ish order
    import random
    shuffled = positions.copy()
    random.shuffle(shuffled)
    shuffled_results = {pos: should_typo_fn(pos, seed, intensity) for pos in shuffled}
    
    all_match = all(
        forward[pos] == reverse[pos] == shuffled_results[pos]
        for pos in positions
    )
    
    if all_match:
        return True, "✓ Order independent: results match regardless of processing order"
    else:
        return False, "✗ Order dependent: results vary with processing order"


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    print("Zerobytes Typo Hash - Self Test")
    print("=" * 50)
    
    # Test determinism
    test_seed = 42
    test_positions = list(range(100))
    
    print("\n1. Position Hash Determinism:")
    hashes_run1 = [position_hash(p, test_seed, LOCATION_SALT) for p in test_positions]
    hashes_run2 = [position_hash(p, test_seed, LOCATION_SALT) for p in test_positions]
    print(f"   Run 1 == Run 2: {hashes_run1 == hashes_run2}")
    
    print("\n2. Should-Typo Determinism:")
    intensity = 0.05
    typos_run1 = [should_typo_at(p, test_seed, intensity) for p in test_positions]
    typos_run2 = [should_typo_at(p, test_seed, intensity) for p in test_positions]
    print(f"   Run 1 == Run 2: {typos_run1 == typos_run2}")
    print(f"   Typo count: {sum(typos_run1)} / {len(test_positions)}")
    
    print("\n3. Order Independence:")
    success, msg = verify_order_independence(should_typo_at, test_positions, test_seed, intensity)
    print(f"   {msg}")
    
    print("\n4. Different Seeds Produce Different Results:")
    typos_seed42 = [should_typo_at(p, 42, intensity) for p in test_positions]
    typos_seed43 = [should_typo_at(p, 43, intensity) for p in test_positions]
    print(f"   Seed 42 != Seed 43: {typos_seed42 != typos_seed43}")
    
    print("\n5. Fatigue Intensity Curve:")
    doc_len = 1000
    for pos in [0, 250, 500, 750, 999]:
        fi = fatigue_intensity(pos, doc_len, test_seed, 0.05)
        print(f"   Position {pos:4d}: intensity = {fi:.4f}")
    
    print("\n" + "=" * 50)
    print("All tests passed!" if all([
        hashes_run1 == hashes_run2,
        typos_run1 == typos_run2,
        success,
        typos_seed42 != typos_seed43
    ]) else "Some tests failed!")
