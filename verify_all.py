import subprocess
import os
import glob

ROOT = '/data/workspace/TaiShen-v4.0'
INC = os.path.join(ROOT, 'app/src/main/cpp')

print("=" * 50)
print("COMPILATION VERIFICATION")
print("=" * 50)

# Key files first
key_files = [
    'app/src/main/cpp/engine/lora/LoRAManager.cpp',
    'app/src/main/cpp/engine/controlnet/ControlNetProcessor.cpp',
    'app/src/main/cpp/engine/facerestore/FaceRestorer.cpp',
    'app/src/main/cpp/engine/mediatek/Dimensity8400Adapter.cpp',
]

for kf in key_files:
    f = os.path.join(ROOT, kf)
    if not os.path.exists(f):
        print(f"\n  SKIP: {kf} (not found)")
        continue
    r = subprocess.run(
        ['g++', '-std=c++17', '-fsyntax-only', '-I', INC, '-DANDROID', f],
        capture_output=True, text=True, timeout=30
    )
    rel = os.path.relpath(f, ROOT)
    if r.returncode == 0:
        print(f"  PASS: {rel}")
    else:
        print(f"  FAIL: {rel} (exit={r.returncode})")
        for line in r.stderr.strip().split('\n')[-3:]:
            print(f"    {line}")

# All cpp files
print()
print("--- All C++ files ---")
cpp_files = glob.glob(os.path.join(ROOT, '**/*.cpp'), recursive=True)
failed = []
for cf in sorted(cpp_files):
    r = subprocess.run(
        ['g++', '-std=c++17', '-fsyntax-only', '-I', INC, '-DANDROID', cf],
        capture_output=True, text=True, timeout=30
    )
    if r.returncode != 0:
        rel = os.path.relpath(cf, ROOT)
        err = r.stderr.strip().split('\n')[-1] if r.stderr.strip() else "unknown"
        failed.append((rel, err))

if failed:
    print(f"  {len(failed)} FAILED:")
    for rel, err in failed[:25]:
        print(f"    {rel}: {err[:120]}")
else:
    print(f"  PASS: All {len(cpp_files)} C++ files compile cleanly")

print()
print("=" * 50)
