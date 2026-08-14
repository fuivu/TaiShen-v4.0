#!/usr/bin/env python3
"""Check namespace brace matching."""
f = 'app/src/main/cpp/engine/mediatek/Dimensity8400Adapter.h'
lines = open(f, encoding='utf-8', errors='ignore').readlines()

# Find namespace lines
for i, line in enumerate(lines, 1):
    s = line.strip()
    if 'namespace' in s:
        print("Line {}: {}".format(i, s[:80]))
    if '{' in s and 'namespace' not in s and 'struct' not in s and 'class' not in s and 'enum' not in s:
        pass  # could be many things

# Count all braces in raw file
raw = ''.join(lines)
print("\nRaw brace count: {}={}, {}={}".format(
    raw.count('{'), raw.count('}'), 
    raw.count('{') - raw.count('}'), 0))
