#!/bin/bash
# ============================================================
#  build-apk.sh — 一键编译 Local AI Painter APK
#  用法: ./build-apk.sh [debug|release]
# ============================================================

set -e

cd "$(dirname "$0")"

MODE=${1:-release}
echo "============================================"
echo "  Local AI Painter v4.0.0 — Build Script"
echo "  Mode: $MODE"
echo "============================================"
echo ""

# 1. 检查 gradlew
if [ ! -x "./gradlew" ]; then
    echo "❌ gradlew 不存在或不可执行"
    echo "   请先运行: bash install-wrapper.sh"
    exit 1
fi

# 2. 检查 gradle-wrapper.jar
if [ ! -f "./gradle/wrapper/gradle-wrapper.jar" ] || [ ! -s "./gradle/wrapper/gradle-wrapper.jar" ]; then
    echo "❌ gradle-wrapper.jar 缺失或为空"
    echo "   请先运行: bash install-wrapper.sh"
    exit 1
fi

# 3. 检查 NDK
if [ -z "$ANDROID_NDK_HOME" ] && [ -z "$ANDROID_NDK_ROOT" ]; then
    echo "⚠️  ANDROID_NDK_HOME 未设置"
    echo "   尝试查找 NDK..."
    # Termux 常见路径
    for p in "$HOME/Android/ndk" "/data/data/com.termux/files/home/Android/ndk" "$PREFIX/opt/android-ndk"; do
        if [ -d "$p" ]; then
            export ANDROID_NDK_HOME="$p"
            echo "   找到: $p"
            break
        fi
    done
fi

# 4. 编译
echo ""
echo "🔨 开始编译..."
echo ""

if [ "$MODE" = "debug" ]; then
    ./gradlew assembleDebug
    APK_PATH="app/build/outputs/apk/debug/app-debug.apk"
else
    ./gradlew assembleRelease
    APK_PATH="app/build/outputs/apk/release/app-release.apk"
fi

# 5. 结果
echo ""
echo "============================================"
if [ -f "$APK_PATH" ]; then
    SIZE=$(stat -c%s "$APK_PATH" 2>/dev/null || stat -f%z "$APK_PATH" 2>/dev/null)
    echo "✅ 编译成功！"
    echo ""
    echo "   APK: $APK_PATH"
    echo "   大小: $((SIZE / 1024 / 1024))MB"
    echo ""
    echo "   安装到手机: adb install -r \"$APK_PATH\""
else
    echo "❌ APK 未生成，请检查编译日志"
    exit 1
fi
echo "============================================"
