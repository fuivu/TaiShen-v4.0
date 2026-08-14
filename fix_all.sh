#!/bin/bash
# 批量修复脚本：统一版本号 + 清理占位符注释
set -e

ROOT="/data/workspace/TaiShen-v4.0"

echo "=== [1/5] 修复构建脚本版本号 ==="

# build-apk.ps1
sed -i 's/Local AI Painter v3\.5\.0 — Windows 一键编译脚本 (PowerShell)/Local AI Painter v4.0.0 — Windows 一键编译脚本 (PowerShell)/g' "$ROOT/build-apk.ps1"
sed -i 's/Local AI Painter v3\.5\.0 Aurora Build/Local AI Painter v4.0.0 TaiShen Build/g' "$ROOT/build-apk.ps1"
sed -i 's/LocalAIPainter-v3\.5\.0\.0-/TaiShen-v4.0.0-/g' "$ROOT/build-apk.ps1"
sed -i 's/Local AI Painter v3\.5\.0 Aurora 构建完成/Local AI Painter v4.0.0 TaiShen 构建完成/g' "$ROOT/build-apk.ps1"

# build-apk.sh
sed -i 's/Local AI Painter v3\.7 — Build Script/Local AI Painter v4.0.0 — Build Script/g' "$ROOT/build-apk.sh"

# install-all.sh
sed -i 's/Local AI Painter v3\.7 一键安装脚本/Local AI Painter v4.0.0 一键安装脚本/g' "$ROOT/install-all.sh"
sed -i 's/Local AI Painter v3\.7 — 一键安装向导/Local AI Painter v4.0.0 — 一键安装向导/g' "$ROOT/install-all.sh"

echo "=== [2/5] 修复 C++ 源码版本号 (v3.5 Aurora → v4.0 TaiShen) ==="

# 批量替换所有 cpp/h 文件中的 v3.5 Aurora → v4.0 TaiShen
find "$ROOT" \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's/v3\.5 Aurora/v4.0 TaiShen/g' {} \;
find "$ROOT" \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's/3\.5\.0 Aurora/4.0.0 TaiShen/g' {} \;
find "$ROOT" \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's/3\.5-aurora/4.0-taishen/g' {} \;
find "$ROOT" \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's/v3\.5 "Aurora"/v4.0 "TaiShen"/g' {} \;
find "$ROOT" \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's/"Aurora"/"TaiShen"/g' {} \;
find "$ROOT" \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's/Aurora 引擎/Aurora 引擎 (legacy naming)/g' {} \;

echo "=== [3/5] 修复 ARCHITECTURE.md 中的旧版本引用 ==="
sed -i 's/v3\.5 Aurora/v4.0 TaiShen/g' "$ROOT/ARCHITECTURE.md"
sed -i 's/Aurora/v4.0 TaiShen/g' "$ROOT/ARCHITECTURE.md"

echo "=== [4/5] 验证修复结果 ==="

echo "--- 检查残留 v3.x / v3\.x ---"
grep -r "v3\." --include="*.kts" --include="*.sh" --include="*.ps1" --include="*.yml" --include="*.md" --include="*.properties" --include="*.toml" "$ROOT" 2>/dev/null | head -10 || echo "✅ 无残留"

echo ""
echo "--- 检查 C++ 残留 v3.5 / Aurora ---"
grep -r "v3\.5\|Aurora\|3\.5\.0" --include="*.cpp" --include="*.h" "$ROOT" 2>/dev/null | head -10 || echo "✅ 无残留"

echo ""
echo "=== [5/5] 检查 versionName / versionCode ==="
grep -r "versionName\|versionCode" "$ROOT/app/build.gradle.kts" 2>/dev/null

echo ""
echo "✅ 全部修复完成"
