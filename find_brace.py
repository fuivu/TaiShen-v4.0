#!/usr/bin/env python3
"""Find exact line with brace mismatch."""
import re

f = 'app/src/main/cpp/engine/mediatek/Dimensity8400Adapter.h'
s = open(f, encoding='utf-8', errors='ignore').read()

# Remove comments and strings
s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
s = re.sub(r'//[^\n]*', '', s)
s = re.sub(r'"(?:\\.|[^"\\])*"', '', s)
s = re.sub(r"'(?:\\.|[^'\\])*'", '', s)

lines = s.split('\n')
depth = 0
for i, line in enumerate(lines, 1):
    for c in line:
        if c == '{': depth += 1
        elif c == '}': depth -= 1
    if depth < 0:
        print(f"Line {i}: depth went negative: {line.strip()[:80]}")
        break
    if i % 50 == 0:
        print(f"Line {i}: depth={depth}")

print(f"\nFinal depth: {depth}")
print(f"Total lines: {len(lines)}")

# Also check: count opening vs closing in raw
raw_open = s.count('{')
raw_close = s.count('}')
print(f"Raw {{ count: {raw_open}, }} count: {raw_close}, diff: {raw_open - raw_close}")
