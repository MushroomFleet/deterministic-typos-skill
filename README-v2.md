# Added-Typos Skill (v2 - Zerobytes Enhanced)

A Claude skill for transforming sterile AI-generated text into authentically human-typed output using **deterministic seed-based typo generation**.

## What's New in v2

**Zerobytes Integration**: Position-as-seed methodology ensures perfect reproducibility.

| Feature | v1 | v2 |
|---------|----|----|
| Typo placement | Random | Deterministic (seeded) |
| Reproducibility | No | Yes—same seed = same output |
| Cross-machine consistency | No | Yes |
| Dual-seed control | No | Yes (position + type seeds) |
| Fatigue profile | Quartile steps | Coherent noise (smooth) |

## Core Principle

> **The position springs the error.** Character index + seed = reproducible typo decisions.

```
Same document + same seed = identical typos
Every. Single. Time.
```

This mimics how humans consistently make the same mistakes—a hallmark of authentic typing patterns.

## Installation

Download `added-typos.skill` and add it to your Claude skills directory:
```
/mnt/skills/user/added-typos/
```

### Dependencies

The Python scripts require `xxhash` for optimal performance:
```bash
pip install xxhash
```

Falls back to `hashlib.md5` if xxhash unavailable (slower but still deterministic).

## Usage

### Basic (Auto-Seeded)

```
"add typos to draft.md"
```

Automatically seeds from document content hash.

### With Explicit Seed (Reproducible)

```
"add typos to draft.md with seed 42"
```

Run this command 100 times—you'll get identical output every time.

### Full Configuration

```
"add typos to report.md with seed 12345, heavy intensity, speed-typist profile"
```

### Advanced: Dual-Seed Mode

```
"add typos with position seed 100 and type seed 200"
```

Independent control over WHERE typos occur vs WHAT typos occur.

## Configuration Options

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `seed` | integer | hash(document) | World seed for all decisions |
| `position_seed` | integer | inherits seed | Override: controls WHERE |
| `type_seed` | integer | inherits seed | Override: controls WHAT |
| `intensity` | `light`, `moderate`, `heavy`, % | `moderate` | Error density |
| `profile` | see below | `balanced` | Pattern distribution |
| `dyslexic` | `true`/`false` | `false` | Enable dyslexic patterns |
| `punctuation` | `true`/`false` | `false` | Enable punctuation habits |

### Intensity Levels

| Level | Error Rate | Use Case |
|-------|------------|----------|
| `light` | 1-2% | Subtle imperfections |
| `moderate` | 3-5% | Natural human typing |
| `heavy` | 7-10% | Fatigued or rushed typing |

### Profiles

| Profile | Dominant Patterns | Best For |
|---------|-------------------|----------|
| `balanced` | Even mix | General authenticity |
| `keyboard-hardware` | Sticky keys (60%) | Worn keyboard simulation |
| `speed-typist` | Adjacent keys (50%) | Fast, careless typing |
| `fatigue` | Progressive increase | Long documents |

## Determinism Demonstration

```python
# Run 1
apply_typos("The quick brown fox", seed=42)
# Output: "Teh quikc brown fox"

# Run 2 (same seed)
apply_typos("The quick brown fox", seed=42)
# Output: "Teh quikc brown fox"  ← IDENTICAL

# Run 3 (different seed)
apply_typos("The quick brown fox", seed=43)
# Output: "The qiuck brwon fox"  ← Different
```

## Command Line Usage

```bash
# Basic usage
python scripts/apply_typos.py input.txt --seed 42

# Full options
python scripts/apply_typos.py input.txt \
    --seed 42 \
    --intensity heavy \
    --profile speed-typist \
    --output output.txt

# Dual seed mode
python scripts/apply_typos.py input.txt \
    --position-seed 100 \
    --type-seed 200
```

## How It Works

### Hash Architecture

```
User Seed (or document hash)
    │
    └─▶ World Seed
           ├─▶ LOCATION_SALT (0x1000) → Should typo exist here?
           ├─▶ TYPE_SALT (0x2000)     → Which pattern?
           ├─▶ VARIANT_SALT (0x3000)  → Which specific variant?
           └─▶ INTENSITY_SALT (0x4000)→ Fatigue curve noise
```

### Decision Flow (Per Character)

```
char_index + seed
        │
        ├─▶ xxhash32(char_index, seed ^ LOCATION_SALT)
        │         │
        │         └─▶ hash_to_float() < intensity? 
        │                   YES → continue
        │                   NO  → skip this position
        │
        ├─▶ xxhash32(char_index, seed ^ TYPE_SALT)
        │         │
        │         └─▶ Weighted selection from profile
        │             Returns: sticky_key | adjacent_key | etc.
        │
        └─▶ xxhash32(char_index, seed ^ VARIANT_SALT)
                  │
                  └─▶ Select specific variant
                      e.g., which adjacent key, double or drop
```

## Preservation Rules

These elements are **never modified**:

- URLs (`https://...`)
- Email addresses
- Code blocks (fenced and inline)
- Numbers and dates
- Markdown syntax

## File Structure

```
added-typos/
├── SKILL.md                        # Claude skill definition
├── README.md                       # This file
├── scripts/
│   ├── typo_hash.py               # Core zerobytes hash functions
│   └── apply_typos.py             # Main application script
└── references/
    ├── qwerty-mapping.md          # Keyboard layout data
    ├── pattern-profiles.md        # Profile definitions
    ├── dyslexic-patterns.md       # Opt-in dyslexic simulation
    └── punctuation-habits.md      # Opt-in punctuation degradation
```

## Verification

Test determinism with the built-in verification:

```bash
python scripts/typo_hash.py
```

Output:
```
Zerobytes Typo Hash - Self Test
==================================================

1. Position Hash Determinism:
   Run 1 == Run 2: True

2. Should-Typo Determinism:
   Run 1 == Run 2: True
   Typo count: 5 / 100

3. Order Independence:
   ✓ Order independent: results match regardless of processing order

4. Different Seeds Produce Different Results:
   Seed 42 != Seed 43: True

5. Fatigue Intensity Curve:
   Position    0: intensity = 0.0150
   Position  250: intensity = 0.0294
   Position  500: intensity = 0.0512
   Position  750: intensity = 0.0847
   Position  999: intensity = 0.1245

==================================================
All tests passed!
```

## Use Cases

- **Reproducible testing**: Generate consistent typo patterns for test suites
- **Version control**: Track exact typo patterns with seed values
- **A/B testing**: Compare different seeds/profiles with reproducible baselines
- **Authenticity**: Human-like consistent mistake patterns
- **Training data**: Generate deterministic datasets with controlled error rates

## Why Determinism Matters

Humans don't make random mistakes—they have consistent blind spots:

- Always misspelling "receive" as "recieve"
- Consistently hitting 'r' instead of 'e'
- Never catching certain typos

Seeded determinism captures this consistency. A document processed with seed `42` will always have the same "mistake fingerprint"—just like a real human typist.

## License

MIT License - See LICENSE file for details.

---

## 📚 Citation

### Academic Citation

If you use this codebase in your research or project, please cite:

```bibtex
@software{added_typos_skill,
  title = {Added-Typos Skill: Deterministic Human Typing Error Simulation for Claude},
  author = {Drift Johnson},
  year = {2025},
  url = {https://github.com/MushroomFleet/deterministic-typos-skill},
  version = {2.0.0}
}
```

### Donate

[![Ko-Fi](https://cdn.ko-fi.com/cdn/kofi3.png?v=3)](https://ko-fi.com/driftjohnson)
