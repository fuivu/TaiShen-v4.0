import subprocess
import os
import glob

ROOT = '/data/workspace/TaiShen-v4.0'

print("=" * 60)
print("COMPILATION VERIFICATION")
print("=" * 60)

# Test 1: LoRAManager.cpp syntax check
print("\n--- Test 1: LoRAManager.cpp syntax check ---")
f = os.path.join(ROOT, 'app/src/main/cpp/engine/lora/LoRAManager.cpp')
cmd = ['g++', '-std=c++17', '-fsyntax-only', '-I', os.path.join(ROOT, 'app/src/main/cpp'),
       '-DANDROID', f]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode == 0:
    print("  PASS: LoRAManager.cpp compiles cleanly")
else:
    print(f"  FAIL (exit={r.returncode})")
    if r.stderr: print(f"  STDERR: {r.stderr[:500]}")

# Test 2: ControlNetProcessor.cpp
print("\n--- Test 2: ControlNetProcessor.cpp syntax check ---")
f = os.path.join(ROOT, 'app/src/main/cpp/engine/controlnet/ControlNetProcessor.cpp')
cmd = ['g++', '-std=c++17', '-fsyntax-only', '-I', os.path.join(ROOT, 'app/src/main/cpp'),
       '-DANDROID', f]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode == 0:
    print("  PASS: ControlNetProcessor.cpp compiles cleanly")
else:
    print(f"  FAIL (exit={r.returncode})")
    if r.stderr: print(f"  STDERR: {r.stderr[:500]}")

# Test 3: FaceRestorer.cpp
print("\n--- Test 3: FaceRestorer.cpp syntax check ---")
f = os.path.join(ROOT, 'app/src/main/cpp/engine/facerestore/FaceRestorer.cpp')
cmd = ['g++', '-std=c++17', '-fsyntax-only', '-I', os.path.join(ROOT, 'app/src/main/cpp'),
       '-DANDROID', f]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode == 0:
    print("  PASS: FaceRestorer.cpp compiles cleanly")
else:
    print(f"  FAIL (exit={r.returncode})")
    if r.stderr: print(f"  STDERR: {r.stderr[:500]}")

# Test 4: Dimensity8400Adapter.cpp
print("\n--- Test 4: Dimensity8400Adapter.cpp syntax check ---")
f = os.path.join(ROOT, 'app/src/main/cpp/engine/mediatek/Dimensity8400Adapter.cpp')
cmd = ['g++', '-std=c++17', '-fsyntax-only', '-I', os.path.join(ROOT, 'app/src/main/cpp'),
       '-DANDROID', f]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode == 0:
    print("  PASS: Dimensity8400Adapter.cpp compiles cleanly")
else:
    print(f"  FAIL (exit={r.returncode})")
    if r.stderr: print(f"  STDERR: {r.stderr[:500]}")

# Test 5: All cpp files syntax check
print("\n--- Test 5: All C++ files syntax check ---")
cpp_files = glob.glob(os.path.join(ROOT, '**/*.cpp'), recursive=True)
inc_dir = os.path.join(ROOT, 'app/src/main/cpp')
failed = []
for cf in sorted(cpp_files):
    cmd = ['g++', '-std=c++17', '-fsyntax-only', '-I', inc_dir, '-DANDROID', cf]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        rel = os.path.relpath(cf, ROOT)
        err = r.stderr.strip().split('\n')[-1] if r.stderr.strip() else "unknown error"
        failed.append((rel, err))

if failed:
    print(f"  {len(failed)} files FAILED:")
    for rel, err in failed[:20]:
        print(f"    ❌ {rel}: {err[:100]}")
else:
    print(f"  PASS: All {len(cpp_files)} C++ files compile cleanly")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
