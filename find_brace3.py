#!/usr/bin/env python3
"""Precise brace tracker - no f-strings with braces."""
import re

f = 'app/src/main/cpp/engine/mediatek/Dimensity8400Adapter.h'
s = open(f, encoding='utf-8', errors='ignore').read()

i = 0
depth = 0
line = 1
in_block = False
in_linec = False
in_dstr = False
in_sstr = False
brace_lines = []

while i < len(s):
    c = s[i]
    nxt = s[i+1] if i+1 < len(s) else ''
    
    if c == '\n':
        line += 1
        in_linec = False
        i += 1
        continue
    
    if in_block:
        if c == '*' and nxt == '/':
            in_block = False
            i += 2
        else:
            i += 1
        continue
    
    if in_linec:
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
    
    if c == '/' and nxt == '*':
        in_block = True
        i += 2
        continue
    if c == '/' and nxt == '/':
        in_linec = True
        i += 2
        continue
    if c == '"':
        in_dstr = True
        i += 1
        continue
    if c == "'":
        in_sstr = True
        i += 1
        continue
    
    if c == '{':
        depth += 1
        brace_lines.append(line)
    elif c == '}':
        depth -= 1
        if depth < 0:
            print("EXTRA } at line " + str(line) + ", depth=" + str(depth))
            if brace_lines:
                print("  Last unmatched { was at line " + str(brace_lines[-1]))
            break
        brace_lines.pop()
    
    i += 1

if depth == 0:
    print("All braces balanced")
else:
    print("Unbalanced: depth=" + str(depth))
    print("Unmatched { at: " + str(brace_lines))
