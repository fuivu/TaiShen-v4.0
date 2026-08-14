#!/usr/bin/env python3
"""Precise brace tracker with char-by-char state machine."""
import re

f = 'app/src/main/cpp/engine/mediatek/Dimensity8400Adapter.h'
s = open(f, encoding='utf-8', errors='ignore').read()

# State machine: track strings, comments, and braces
i = 0
depth = 0
line = 1
in_sline = False  # /* */
in_mline = False  # //
in_dstr = False   # "..."  
in_sstr = False   # '...'
brace_line = []  # track where each { is

while i < len(s):
    c = s[i]
    nxt = s[i+1] if i+1 < len(s) else ''
    
    if c == '\n':
        line += 1
        in_sline = False
        i += 1
        continue
    
    if in_sline:
        if c == '*' and nxt == '/':
            in_sline = False
            i += 2
        else:
            i += 1
        continue
    
    if in_mline:
        i += 1
        continue
    
    if in_dstr:
        if c == '\\' and nxt:
            i += 2
            continue
        if c == '"':
            in_dstr = False
        i += 1
        continue
    
    if in_sstr:
        if c == '\\' and nxt:
            i += 2
            continue
        if c == "'":
            in_sstr = False
        i += 1
        continue
    
    # Check comment starts
    if c == '/' and nxt == '*':
        in_sline = True
        i += 2
        continue
    if c == '/' and nxt == '/':
        in_mline = True
        i += 2
        continue
    
    # Check string starts
    if c == '"':
        in_dstr = True
        i += 1
        continue
    if c == "'":
        in_sstr = True
        i += 1
        continue
    
    # Track braces
    if c == '{':
        depth += 1
        brace_line.append(line)
    elif c == '}':
        depth -= 1
        if depth < 0:
            print(f"❌ EXTRA } at line {line}, depth={depth}")
            # Find matching { 
            if brace_line:
                print(f"   Last unmatched {{ was at line {brace_line[-1]}")
            break
        brace_line.pop()
    
    i += 1

if depth == 0:
    print("✅ All braces balanced")
else:
    print(f"❌ Unbalanced: depth={depth}")
    if brace_line:
        print(f"   Unmatched {{ at lines: {brace_line}")
