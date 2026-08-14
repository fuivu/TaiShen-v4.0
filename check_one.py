#!/usr/bin/env python3
"""Compile-check C++ files one by one."""
import subprocess, glob, os

src_dir = 'app/src/main/cpp'
includes = '-I{} -I{}/engine -I{}/v35'.format(src_dir, src_dir, src_dir)

cpp_files = sorted(glob.glob('{}//**/*.cpp'.format(src_dir), recursive=True))

pass_c = warn_c = fail_c = 0
fails = []

for f in cpp_files:
    ret = subprocess.run(
        ['g++', '-std=c++17', '-fsyntax-only'] + 
        includes.split() + [f],
        capture_output=True, text=True
    )
    out = (ret.stdout + ret.stderr).strip()
    
    if not out:
        pass_c += 1
        print("  OK: " + os.path.relpath(f))
        continue
    
    lines = [l for l in out.split('\n') if l.strip()]
    real = [l for l in lines if 'No such file' not in l]
    
    if len(real) == 0:
        warn_c += 1
        print("  WARN: " + os.path.relpath(f) + " (NDK headers)")
    else:
        fail_c += 1
        print("  FAIL: " + os.path.relpath(f))
        for l in real[:3]:
            print("    " + l.strip()[:120])
        fails.append(f)

print("\n" + "="*50)
print("Pass: {}  Warn: {}  Fail: {}".format(pass_c, warn_c, fail_c))
print("Total: {}".format(len(cpp_files)))

if fails:
    print("\nFailed:")
    for f in fails:
        print("  " + f)
