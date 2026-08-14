#!/usr/bin/env python3
"""Correct brace verifier - handles regex special chars properly."""
import re, glob

files = (glob.glob('app/src/main/cpp/**/*.cpp', recursive=True) +
         glob.glob('app/src/main/cpp/**/*.h', recursive=True) +
         glob.glob('engine/**/*.cpp', recursive=True) +
         glob.glob('engine/**/*.h', recursive=True))

for f in sorted(files):
    try:
        s = open(f, encoding='utf-8', errors='ignore').read()
    except:
        continue
    
    # Remove block comments
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
    # Remove line comments
    s = re.sub(r'//[^\n]*', '', s)
    # Remove string literals (including escaped quotes)
    s = re.sub(r'"(?:\\.|[^"\\])*"', '', s)
    s = re.sub(r"'(?:\\.|[^'\\])*'", '', s)
    
    cb = s.count('{') - s.count('}')
    pb = s.count('(') - s.count(')')
    sb = s.count('[') - s.count(']')
    
    if cb or pb or sb:
        parts = []
        if cb: parts.append('{ %+d' % cb)
        if pb: parts.append('( %+d' % pb)
        if sb: parts.append('[ %+d' % sb)
        print(f"  {f}: {', '.join(parts)}")

print(f"\nChecked {len(files)} files")
