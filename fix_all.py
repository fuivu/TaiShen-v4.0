#!/usr/bin/env python3
"""批量修复版本号和占位符注释"""
import os
import re
import glob

ROOT = "/data/workspace/TaiShen-v4.0"

# ========== 文件收集 ==========
def collect_files(patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(ROOT, pat), recursive=True))
    return sorted(set(files))

# ========== 替换函数 ==========
def replace_in_file(filepath, replacements):
    """replacements: list of (old, new)"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    modified = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            modified = True
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    return modified

# ========== 1. 构建脚本 ==========
print("=== [1/5] 修复构建脚本版本号 ===")
script_files = collect_files(["build-apk.ps1", "build-apk.sh", "install-all.sh",
                               "install-neuron-so.sh", "install-wrapper-jar.sh",
                               "setup-git-repo.sh", "fetch-wrapper-jar.sh"])
for f in script_files:
    changes = [
        ("Local AI Painter v3.5.0", "Local AI Painter v4.0.0"),
        ("Local AI Painter v3.5.0 Aurora Build", "Local AI Painter v4.0.0 TaiShen Build"),
        ("LocalAIPainter-v3.5.0.0-", "TaiShen-v4.0.0-"),
        ("Local AI Painter v3.5.0 Aurora 构建完成", "Local AI Painter v4.0.0 TaiShen 构建完成"),
        ("Local AI Painter v3.7", "Local AI Painter v4.0.0"),
        ("v3.7 — Build Script", "v4.0.0 — Build Script"),
        ("v3.7 一键安装脚本", "v4.0.0 一键安装脚本"),
        ("v3.7 — 一键安装向导", "v4.0.0 — 一键安装向导"),
        ("v3.0", "v4.0.0"),
    ]
    if replace_in_file(f, changes):
        print(f"  ✅ {os.path.relpath(f, ROOT)}")

# ========== 2. C++ 源码版本号 ==========
print("\n=== [2/5] 修复 C++ 源码版本号 ===")
cpp_files = collect_files(["**/*.cpp", "**/*.h"])
cpp_replacements = [
    ("v3.5 Aurora", "v4.0 TaiShen"),
    ("3.5.0 Aurora", "4.0.0 TaiShen"),
    ("3.5-aurora", "4.0-taishen"),
    ('v3.5 "Aurora"', 'v4.0 "TaiShen"'),
    ("Aurora 引擎", "TaiShen 引擎"),
    ("aurora", "taishen"),
    ("AURORA", "TAISHEN"),
]
count = 0
for f in cpp_files:
    if replace_in_file(f, cpp_replacements):
        count += 1
        print(f"  ✅ {os.path.relpath(f, ROOT)}")
print(f"  共修复 {count} 个 C++ 文件")

# ========== 3. Markdown 文档 ==========
print("\n=== [3/5] 修复文档版本号 ===")
md_files = collect_files(["**/*.md"])
md_replacements = [
    ("v3.5 Aurora", "v4.0 TaiShen"),
    ("v3.7", "v4.0.0"),
    ("3.5.0", "4.0.0"),
]
for f in md_files:
    if replace_in_file(f, md_replacements):
        print(f"  ✅ {os.path.relpath(f, ROOT)}")

# ========== 4. YAML / CI ==========
print("\n=== [4/5] 修复 CI/CD 配置 ===")
ci_files = collect_files([".github/**/*.yml", "**/*.yml", "**/*.yaml"])
ci_replacements = [
    ("v3.5 Aurora", "v4.0 TaiShen"),
    ("v3.7", "v4.0.0"),
    ("3.5.0", "4.0.0"),
    ("gradle-9.3.1", "gradle-8.7"),
    ("GRADLE_VERSION: 9.3.1", "GRADLE_VERSION: 8.7"),
]
for f in ci_files:
    if replace_in_file(f, ci_replacements):
        print(f"  ✅ {os.path.relpath(f, ROOT)}")

# ========== 5. 验证 ==========
print("\n=== [5/5] 验证结果 ===")

# 检查残留 v3
print("\n--- 残留 v3.x 检查 (非注释行) ---")
import subprocess
result = subprocess.run(
    ['grep', '-r', '--include=*.kts', '--include=*.sh', '--include=*.ps1',
     '--include=*.yml', '--include=*.md', '--include=*.properties',
     '--include=*.toml', '-E', 'v3\.[0-9]', ROOT],
    capture_output=True, text=True
)
if result.stdout.strip():
    for line in result.stdout.strip().split('\n')[:15]:
        print(f"  ⚠️  {line}")
else:
    print("  ✅ 零残留")

# 检查 C++ 残留
print("\n--- C++ 残留 v3.5 / Aurora 检查 ---")
result = subprocess.run(
    ['grep', '-r', '--include=*.cpp', '--include=*.h', '-E', 'v3\.5|Aurora|aurora', ROOT],
    capture_output=True, text=True
)
if result.stdout.strip():
    for line in result.stdout.strip().split('\n')[:15]:
        print(f"  ⚠️  {line}")
else:
    print("  ✅ 零残留")

# 检查 versionName / versionCode
print("\n--- app/build.gradle.kts 版本检查 ---")
gradle_kts = os.path.join(ROOT, "app/build.gradle.kts")
if os.path.exists(gradle_kts):
    with open(gradle_kts) as f:
        for line in f:
            if 'versionName' in line or 'versionCode' in line:
                print(f"  {line.strip()}")

print("\n✅ 全部修复完成")
