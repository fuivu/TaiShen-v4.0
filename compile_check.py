#!/usr/bin/env python3
"""Compile-check all C++ files, report results."""
import subprocess, glob, os, sys

src_dir = 'app/src/main/cpp'
includes = f"-I{src_dir} -I{src_dir}/engine -I{src_dir}/v35"

cpp_files = sorted(glob.glob(f'{src_dir}/**/*.cpp', recursive=True))

pass_count = 0
fail_count = 0
warn_count = 0
failures = []

for f in cpp_files:
    result = subprocess.run(
        f"g++ -std=c++17 -fsyntax-only {includes} \"{f}\" 2>&1",
        shell=True, capture_output=True, text=True
    )
    output = result.stdout + result.stderr
    
    if not output.strip():
        pass_count += 1
        print(f"  ✅ {os.path.relpath(f)}")
        continue
    
    # Check if errors are only "No such file" (missing NDK headers)
    lines = [l for l in output.strip().split('\n') if l.strip()]
    real_errors = [l for l in lines 
                   if 'fatal error' not in l or 'No such file' not in l]
    no_such_file = [l for l in lines if 'No such file' in l]
    
    if len(real_errors) == 0 and len(no_such_file) > 0:
        warn_count += 1
        print(f"  ⚠️  {os.path.relpath(f)} (missing NDK headers)")
    else:
        fail_count += 1
        print(f"  ❌ {os.path.relpath(f)}")
        for l in real_errors[:5]:
            print(f"     {l.strip()[:120]}")
        failures.append(f)

print(f"\n{'='*50}")
print(f"✅ Pass: {pass_count}")
print(f"⚠️  Warn (NDK headers): {warn_count}")
print(f"❌ Fail: {fail_count}")
print(f"Total: {len(cpp_files)}")

if failures:
    print(f"\nFailed files:")
    for f in failures:
        print(f"  {f}")
