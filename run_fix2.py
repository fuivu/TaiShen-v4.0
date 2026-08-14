import os
import glob
import re

ROOT = '/data/workspace/TaiShen-v4.0'

def replace_in_file(filepath, replacements):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return False
    modified = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            modified = True
    if modified:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        except:
            pass
    return modified

# === 1. C++ files - comprehensive Aurora/v3.5 cleanup ===
print('=== Fixing C++ files (comprehensive) ===')
cpp_files = []
for ext in ['*.cpp', '*.h']:
    cpp_files.extend(glob.glob(os.path.join(ROOT, '**', ext), recursive=True))

count = 0
for f in cpp_files:
    changes = [
        # Exact patterns from grep results
        ('v3.5 Aurora', 'v4.0 TaiShen'),
        ('3.5.0 Aurora', '4.0.0 TaiShen'),
        ('v3.5 "Aurora"', 'v4.0 "TaiShen"'),
        ('v3.5 "aurora"', 'v4.0 "TaiShen"'),
        # Remaining patterns
        ('v3.5', 'v4.0'),
        ('3.5.0', '4.0.0'),
        ('Aurora 引擎', 'TaiShen 引擎'),
        ('Aurora引擎', 'TaiShen引擎'),
        ('"Aurora"', '"TaiShen"'),
        ("'Aurora'", "'TaiShen'"),
        # Case variations
        ('Aurora', 'TaiShen'),
        ('aurora', 'taishen'),
    ]
    if replace_in_file(f, changes):
        count += 1
        print(f'  OK: {os.path.relpath(f, ROOT)}')
print(f'  Total: {count} C++ files fixed')

# === 2. Architecture.md ===
print()
print('=== Fixing ARCHITECTURE.md ===')
f = os.path.join(ROOT, 'ARCHITECTURE.md')
changes = [
    ('v3.5 Aurora', 'v4.0 TaiShen'),
    ('v3.5', 'v4.0'),
    ('Aurora', 'TaiShen'),
    ('aurora', 'taishen'),
]
if replace_in_file(f, changes):
    print(f'  OK: ARCHITECTURE.md')

# === 3. Verify ===
print()
print('=== Final Verification ===')

import subprocess

# Check 1: Any v3.x in scripts/docs/configs
print('\n--- Check 1: v3.x in non-C++ files ---')
r = subprocess.run(
    ['grep', '-r', '--include=*.kts', '--include=*.sh', '--include=*.ps1',
     '--include=*.yml', '--include=*.md', '--include=*.properties',
     '--include=*.toml', '-E', 'v3\\.[0-9]', ROOT],
    capture_output=True, text=True
)
# Filter out our own fix scripts
skip_files = ['fix_all.sh', 'run_fix.py', 'run_fix2.py', 'fix_all.py']
lines = []
for l in r.stdout.strip().split('\n'):
    skip = False
    for s in skip_files:
        if s in l:
            skip = True
            break
    if not skip and l.strip():
        lines.append(l)
if lines:
    for l in lines[:10]:
        print(f'  WARN: {l[:130]}')
else:
    print('  PASS: zero residual v3.x')

# Check 2: Any Aurora/aurora in C++
print('\n--- Check 2: Aurora in C++ files ---')
r = subprocess.run(
    ['grep', '-r', '--include=*.cpp', '--include=*.h', '-E', 'Aurora|aurora', ROOT],
    capture_output=True, text=True
)
lines = [l for l in r.stdout.strip().split('\n') if l.strip()]
if lines:
    for l in lines[:10]:
        print(f'  WARN: {l[:130]}')
else:
    print('  PASS: zero residual Aurora')

# Check 3: Any v3.5/v4.0 mismatch in C++
print('\n--- Check 3: v3.5 in C++ files ---')
r = subprocess.run(
    ['grep', '-r', '--include=*.cpp', '--include=*.h', 'v3\\.5', ROOT],
    capture_output=True, text=True
)
lines = [l for l in r.stdout.strip().split('\n') if l.strip()]
if lines:
    for l in lines[:10]:
        print(f'  WARN: {l[:130]}')
else:
    print('  PASS: zero residual v3.5')

# Check 4: versionName/versionCode
print('\n--- Check 4: versionName/versionCode ---')
gradle_kts = os.path.join(ROOT, 'app/build.gradle.kts')
if os.path.exists(gradle_kts):
    with open(gradle_kts) as f:
        for line in f:
            if 'versionName' in line or 'versionCode' in line:
                print(f'  {line.strip()}')

# Check 5: Gradle version
print('\n--- Check 5: Gradle wrapper version ---')
gw = os.path.join(ROOT, 'gradle/wrapper/gradle-wrapper.properties')
if os.path.exists(gw):
    with open(gw) as f:
        for line in f:
            if 'distributionUrl' in line or 'gradle-' in line:
                print(f'  {line.strip()}')

print('\nDONE')
