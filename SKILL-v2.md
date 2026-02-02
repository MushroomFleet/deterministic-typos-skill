---
name: added-typos
description: Add authentic human typing errors to text documents with deterministic seed-based reproducibility. Simulates natural typo patterns including sticky keys, adjacent key misses, letter transpositions, and optionally dyslexic spelling or punctuation habits. Uses xxhash32 position-as-seed methodology for perfect reproducibility—same document + same seed = identical typos every time. Triggers on phrases like "using added-typos skill", "add typos to", "amend with added-typos", "simulate typing errors", or when user wants to make text appear naturally human-typed rather than AI-generated. Accepts any text file (.md, .txt, .docx, .html) as input.
---

# Added-Typos Skill (v2 - Zerobytes Deterministic)

## CRITICAL: Script Execution Required

**DO NOT attempt to implement this algorithm mentally.**

Claude's text generation is inherently non-deterministic. To achieve reproducible typos, Claude MUST:

1. Save the input text to a file
2. Execute `scripts/apply_typos.py` with the specified seed
3. Return the script's output verbatim

**This is non-negotiable for determinism.**

## Workflow (Mandatory)

```bash
# Step 1: Save input to temp file
echo "user's text content" > /tmp/input.txt

# Step 2: Execute script with seed
python scripts/apply_typos.py /tmp/input.txt --seed 42 --intensity moderate --profile balanced

# Step 3: Return output exactly as produced
```

## Configuration Parsing

Extract from user prompt:

| Parameter | Values | Default |
|-----------|--------|---------|
| `seed` | integer | auto-generated from content |
| `intensity` | `light`, `moderate`, `heavy`, or % | `moderate` |
| `profile` | `balanced`, `keyboard-hardware`, `speed-typist`, `fatigue` | `balanced` |

**Prompt examples:**
- `"add typos to draft.md"` → use auto-seed
- `"add typos with seed 42"` → use seed 42
- `"amend with added-typos, seed 12345, heavy"` → seed 12345, heavy intensity

## Script Usage

```bash
# Basic (auto-seed from content)
python scripts/apply_typos.py input.txt

# With explicit seed (REPRODUCIBLE)
python scripts/apply_typos.py input.txt --seed 42

# Full options
python scripts/apply_typos.py input.txt \
    --seed 42 \
    --intensity moderate \
    --profile balanced \
    --output output.txt

# Verify determinism
python scripts/apply_typos.py input.txt --seed 42 --verify
```

## Intensity Levels

| Level | Error Rate | Description |
|-------|------------|-------------|
| `light` | ~1.5% | Subtle imperfections |
| `moderate` | ~4% | Natural human typing |
| `heavy` | ~8.5% | Fatigued/rushed typing |

## Profiles

| Profile | Dominant Patterns |
|---------|-------------------|
| `balanced` | Even mix of all error types |
| `keyboard-hardware` | Heavy sticky keys (60%) |
| `speed-typist` | Adjacent key errors (50%) |
| `fatigue` | Progressive increase toward end |

## Preservation (Automatic)

The script automatically preserves:
- URLs and emails
- Code blocks (fenced and inline)
- Numbers and dates
- Quoted strings

## Determinism Guarantee

When using the script:
```
Same input + Same seed = IDENTICAL output
Every. Single. Time.
```

**Why this matters:** Humans consistently make the same typos. Seed-based determinism captures this authentic pattern.

## Example Session

User: "add typos to my draft with seed 42"

Claude's execution:
```bash
# Save user's content
cat > /tmp/draft.txt << 'EOF'
The quick brown fox jumps over the lazy dog.
EOF

# Apply typos deterministically
python scripts/apply_typos.py /tmp/draft.txt --seed 42 --intensity moderate

# Output (ALWAYS identical for seed 42):
# The quikc brown fox jumps over the laxy dog.
```

## Response Format

After applying typos:
1. Show the modified text
2. Report the seed used: `"Applied with seed: 42"`
3. Note: "Use this seed to reproduce identical results"

## Anti-Pattern Warning

**NEVER do this:**
```
# WRONG - Claude trying to "mentally apply" typos
"Let me add some typos... teh quikc brown..."
```

**ALWAYS do this:**
```bash
# CORRECT - Execute the script
python scripts/apply_typos.py input.txt --seed 42
```

Mental implementation = broken determinism = failed skill.
