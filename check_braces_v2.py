#!/usr/bin/env python3
"""Check brace/paren/bracket balance in C++ files, ignoring comments & strings."""
import re, glob, sys

files = (glob.glob('app/src/main/cpp/**/*.cpp', recursive=True) +
         glob.glob('app/src/main/cpp/**/*.h', recursive=True) +
         glob.glob('engine/**/*.cpp', recursive=True) +
         glob.glob('engine/**/*.h', recursive=True) +
         glob.glob('hybrid/**/*.cpp', recursive=True) +
         glob.glob('hybrid/**/*.h', recursive=True) +
         glob.glob('xunji/**/*.cpp', recursive=True) +
         glob.glob('xunji/**/*.h', recursive=True) +
         glob.glob('hunyuan/**/*.cpp', recursive=True) +
         glob.glob('hunyuan/**/*.h', recursive=True))

def strip_comments_and_strings(text):
    """Remove C/C++ comments and string literals."""
    result = []
    i = 0
    in_str = None  # None, '"', "'"
    in_raw = False
    prev = ''
    while i < len(text):
        c = text[i]
        nxt = text[i+1] if i+1 < len(text) else ''
        
        if in_str:
            if c == '\\' and nxt:
                i += 2
                continue
            if c == in_str:
                in_str = None
            i += 1
            continue
        
        if in_raw:
            if c == '"' and nxt == ')':
                in_raw = False
                i += 2
                continue
            i += 1
            continue
        
        # Check raw string R"delimiter(...)delimiter"
        if c == 'R' and nxt == '"':
            # simplified: skip R"..."
            result.append(c)
            i += 1
            continue
        
        # Block comment
        if c == '/' and nxt == '*':
            i += 2
            while i < len(text) and not (text[i] == '*' and i+1 < len(text) and text[i+1] == '/'):
                i += 1
            i += 2
            continue
        
        # Line comment
        if c == '/' and nxt == '/':
            i += 2
            while i < len(text) and text[i] != '\n':
                i += 1
            continue
        
        # String literal
        if c == '"':
            in_str = '"'
            i += 1
            continue
        
        # Char literal
        if c == "'":
            in_str = "'"
            i += 1
            continue
        
        result.append(c)
        i += 1
    
    return ''.join(result)

bad = []
for f in sorted(files):
    try:
        s = open(f, encoding='utf-8', errors='ignore').read()
    except:
        continue
    s2 = strip_comments_and_strings(s)
    cb = s2.count('{') - s2.count('}')
    pb = s2.count('(') - s2.count(')')
    sb = s2.count('[') - s2.count(']')
    if cb or pb or sb:
        bad.append((f, cb, pb, sb))

if bad:
    print(f"❌ Found {len(bad)} files with unbalanced brackets:")
    for f, cb, pb, sb in bad:
        parts = []
        if cb: parts.append(f"{{ {cb:+d}}}")
        if pb: parts.append(f"( {pb:+d})")
        if sb: parts.append(f"[ {sb:+d}]")
        print(f"  {f}: {', '.join(parts)}")
else:
    print("✅ ALL C++ files have balanced brackets")

print(f"\nChecked {len(files)} files total")
