import os
import glob
import subprocess

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

# 1. Script files
print('=== [1/5] Fixing build scripts ===')
for pat in ['build-apk.ps1', 'build-apk.sh', 'install-all.sh',
            'install-neuron-so.sh', 'install-wrapper-jar.sh',
            'setup-git-repo.sh', 'fetch-wrapper-jar.sh']:
    f = os.path.join(ROOT, pat)
    if not os.path.exists(f):
        continue
    changes = [
        ('Local AI Painter v3.5.0', 'Local AI Painter v4.0.0'),
        ('Local AI Painter v3.5.0 Aurora Build', 'Local AI Painter v4.0.0 TaiShen Build'),
        ('LocalAIPainter-v3.5.0.0-', 'TaiShen-v4.0.0-'),
        ('Local AI Painter v3.5.0 Aurora', 'Local AI Painter v4.0.0 TaiShen'),
        ('Local AI Painter v3.7', 'Local AI Painter v4.0.0'),
        ('v3.7 - Build Script', 'v4.0.0 - Build Script'),
        ('v3.7 -- Build Script', 'v4.0.0 -- Build Script'),
        ('v3.7 一键安装脚本', 'v4.0.0 一键安装脚本'),
        ('v3.7 — 一键安装向导', 'v4.0.0 — 一键安装向导'),
        ('v3.0', 'v4.0.0'),
    ]
    if replace_in_file(f, changes):
        print(f'  OK: {os.path.relpath(f, ROOT)}')

# 2. C++ files
print()
print('=== [2/5] Fixing C++ source files ===')
cpp_files = []
for ext in ['*.cpp', '*.h']:
    cpp_files.extend(glob.glob(os.path.join(ROOT, '**', ext), recursive=True))

count = 0
for f in cpp_files:
    changes = [
        ('v3.5 Aurora', 'v4.0 TaiShen'),
        ('3.5.0 Aurora', '4.0.0 TaiShen'),
        ('3.5-aurora', '4.0-taishen'),
        ('v3.5 "Aurora"', 'v4.0 "TaiShen"'),
        ('Aurora 引擎', 'TaiShen 引擎'),
    ]
    if replace_in_file(f, changes):
        count += 1
        print(f'  OK: {os.path.relpath(f, ROOT)}')
print(f'  Total: {count} C++ files fixed')

# 3. Markdown
print()
print('=== [3/5] Fixing Markdown docs ===')
md_files = glob.glob(os.path.join(ROOT, '**', '*.md'), recursive=True)
for f in md_files:
    changes = [
        ('v3.5 Aurora', 'v4.0 TaiShen'),
        ('v3.7', 'v4.0.0'),
        ('3.5.0', '4.0.0'),
    ]
    if replace_in_file(f, changes):
        print(f'  OK: {os.path.relpath(f, ROOT)}')

# 4. CI YAML
print()
print('=== [4/5] Fixing CI/CD config ===')
ci_files = glob.glob(os.path.join(ROOT, '.github', '**', '*.yml'), recursive=True)
for f in ci_files:
    changes = [
        ('v3.5 Aurora', 'v4.0 TaiShen'),
        ('v3.7', 'v4.0.0'),
        ('3.5.0', '4.0.0'),
        ('gradle-9.3.1', 'gradle-8.7'),
        ('GRADLE_VERSION: 9.3.1', 'GRADLE_VERSION: 8.7'),
    ]
    if replace_in_file(f, changes):
        print(f'  OK: {os.path.relpath(f, ROOT)}')

# 5. Verify
print()
print('=== [5/5] Verification ===')
print()
print('--- Residual v3.x check ---')
r = subprocess.run(['grep', '-r', '--include=*.kts', '--include=*.sh',
    '--include=*.ps1', '--include=*.yml', '--include=*.md',
    '--include=*.properties', '--include=*.toml', '-E', 'v3\\.[0-9]', ROOT],
    capture_output=True, text=True)
lines = [l for l in r.stdout.strip().split('\n') if l.strip()]
if lines:
    for l in lines[:15]:
        print(f'  WARN: {l[:130]}')
else:
    print('  OK: zero residual')

print()
print('--- Residual Aurora check (C++) ---')
r = subprocess.run(['grep', '-r', '--include=*.cpp', '--include=*.h',
    '-E', 'Aurora|aurora|v3\\.5', ROOT],
    capture_output=True, text=True)
lines = [l for l in r.stdout.strip().split('\n') if l.strip()]
if lines:
    for l in lines[:15]:
        print(f'  WARN: {l[:130]}')
else:
    print('  OK: zero residual')

print()
print('--- versionName/versionCode check ---')
gradle_kts = os.path.join(ROOT, 'app/build.gradle.kts')
if os.path.exists(gradle_kts):
    with open(gradle_kts) as f:
        for line in f:
            if 'versionName' in line or 'versionCode' in line:
                print(f'  {line.strip()}')

print()
print('DONE')
