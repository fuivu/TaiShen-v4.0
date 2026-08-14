#!/bin/bash
# ============================================================
#  install-all.sh — Local AI Painter v4.0.0 一键安装脚本
#  
#  功能：
#  1. 获取真实 gradle-wrapper.jar（3 种方法）
#  2. 设置 gradlew 可执行权限
#  3. 检查 Java / Android SDK / NDK
#  4. 检测天玑 8400 NPU .so（可选）
#  5. 运行环境验证
#
#  适用：Termux / Linux / macOS
# ============================================================

set -e

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Local AI Painter v4.0.0 — 一键安装向导   ║"
echo "╚══════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PASS=0
WARN=0
FAIL=0

pass() { echo "  ✅ $1"; PASS=$((PASS+1)); }
warn() { echo "  ⚠️  $1"; WARN=$((WARN+1)); }
fail() { echo "  ❌ $1"; FAIL=$((FAIL+1)); }

# ────────────────────────────────────────────────
echo "━━━ 第 1 步：gradle-wrapper.jar ━━━"
echo ""
JAR_PATH="gradle/wrapper/gradle-wrapper.jar"

if [ -f "$JAR_PATH" ]; then
    SIZE=$(stat -c%s "$JAR_PATH" 2>/dev/null || stat -f%z "$JAR_PATH" 2>/dev/null)
    if [ "$SIZE" -gt 1000 ]; then
        pass "gradle-wrapper.jar 已就绪 (${SIZE} bytes)"
    else
        warn "gradle-wrapper.jar 是占位符 (${SIZE} bytes)，尝试获取..."
        echo ""
        if [ -f "fetch-wrapper-jar.sh" ]; then
            bash fetch-wrapper-jar.sh || true
        fi
        # 再检查一次
        if [ -f "$JAR_PATH" ]; then
            SIZE=$(stat -c%s "$JAR_PATH" 2>/dev/null || stat -f%z "$JAR_PATH" 2>/dev/null)
            if [ "$SIZE" -gt 1000 ]; then
                pass "获取成功 (${SIZE} bytes)"
            else
                fail "自动获取失败，请手动放置"
                echo ""
                echo "  💡 手动方法："
                echo "    1. 从 Local Dream 项目的 gradle/wrapper/ 复制"
                echo "    2. 或在电脑上运行: gradle wrapper --gradle-version 9.3.1"
                echo "    3. 或下载 https://services.gradle.org/distributions/gradle-9.3.1-bin.zip"
                echo "       解压后从 lib/plugins/gradle-wrapper-*.jar 中提取"
            fi
        fi
    fi
else
    warn "gradle-wrapper.jar 不存在，尝试获取..."
    mkdir -p "gradle/wrapper"
    if [ -f "fetch-wrapper-jar.sh" ]; then
        bash fetch-wrapper-jar.sh || true
    fi
fi

echo ""

# ────────────────────────────────────────────────
echo "━━━ 第 2 步：gradlew 脚本 ━━━"
echo ""
if [ -f "gradlew" ]; then
    pass "gradlew 存在"
    chmod +x gradlew 2>/dev/null && pass "已设置可执行权限" || warn "无法设置执行权限（可能权限不足）"
else
    fail "gradlew 不存在"
fi

if [ -f "gradlew.bat" ]; then
    pass "gradlew.bat 存在 (Windows)"
else
    warn "gradlew.bat 不存在（仅影响 Windows 用户）"
fi

echo ""

# ────────────────────────────────────────────────
echo "━━━ 第 3 步：Java 环境 ━━━"
echo ""
if command -v java >/dev/null 2>&1; then
    JAVA_VER=$(java -version 2>&1 | head -1)
    pass "Java 已安装: $JAVA_VER"
    # 检查版本是否 >= 17
    MAJOR=$(java -version 2>&1 | head -1 | sed 's/.*"\([0-9]*\).*"/\1/')
    if [ "$MAJOR" -ge 17 ] 2>/dev/null; then
        pass "Java 版本满足要求 (>= 17)"
    else
        warn "Java 版本可能过低，推荐 17+"
    fi
else
    warn "Java 未安装"
    echo "  💡 Termux: pkg install openjdk-17"
    echo "  💡 Ubuntu: apt install openjdk-17-jdk"
    echo "  💡 macOS: brew install openjdk@17"
fi

echo ""

# ────────────────────────────────────────────────
echo "━━━ 第 4 步：Android SDK / NDK ━━━"
echo ""
if [ -n "$ANDROID_HOME" ] || [ -n "$ANDROID_SDK_ROOT" ]; then
    SDK_DIR="${ANDROID_HOME:-$ANDROID_SDK_ROOT}"
    pass "ANDROID_HOME=$SDK_DIR"
    
    # 检查 NDK
    if [ -d "$SDK_DIR/ndk" ]; then
        NDK_VER=$(ls "$SDK_DIR/ndk/" 2>/dev/null | head -1)
        if [ -n "$NDK_VER" ]; then
            pass "NDK 已安装: $NDK_VER"
            export ANDROID_NDK_HOME="$SDK_DIR/ndk/$NDK_VER"
        fi
    elif [ -n "$ANDROID_NDK_HOME" ]; then
        pass "ANDROID_NDK_HOME=$ANDROID_NDK_HOME"
    else
        warn "NDK 未找到，请安装 NDK 25+"
        echo "  💡 sdkmanager \"ndk;25.2.9519653\""
    fi
    
    # 检查 build-tools
    if [ -d "$SDK_DIR/build-tools" ]; then
        BT_VER=$(ls "$SDK_DIR/build-tools/" 2>/dev/null | head -1)
        [ -n "$BT_VER" ] && pass "Build-Tools: $BT_VER" || warn "Build-Tools 未安装"
    fi
else
    warn "ANDROID_HOME 未设置"
    echo "  💡 Termux: pkg install android-sdk android-ndk"
    echo "  💡 或设置: export ANDROID_HOME=/path/to/sdk""
fi

echo ""

# ────────────────────────────────────────────────
echo "━━━ 第 5 步：天玑 8400 NPU (.so) ━━━"
echo ""
SO_DIR="so-input"
if [ -d "$SO_DIR" ]; then
    SO_COUNT=$(find "$SO_DIR" -name "*.so" | wc -l)
    if [ "$SO_COUNT" -gt 0 ]; then
        pass "找到 ${SO_COUNT} 个 .so 文件"
        find "$SO_COUNT" -name "*.so" -exec echo "     {}" \; 2>/dev/null || true
        if [ -f "install-neuron-so.sh" ]; then
            echo ""
            echo "  💡 运行 bash install-neuron-so.sh 安装到 jniLibs/"
        fi
    else
        warn "$SO_DIR/ 目录为空"
        echo "  💡 从 MediaTek NeuroPilot SDK 提取 .so 放入 $SO_DIR/"
    fi
else
    warn "so-input/ 目录不存在（NPU 加速可选）"
    echo "  💡 没有 .so 也能跑 CPU/Vulkan 模式"
fi

echo ""

# ────────────────────────────────────────────────
echo "━━━ 第 6 步：项目文件完整性 ━━━"
echo ""
check_file() {
    if [ -f "$1" ]; then
        pass "$1"
    else
        fail "$1 缺失"
    fi
}

check_file "build.gradle.kts"
check_file "settings.gradle.kts"
check_file "app/build.gradle.kts"
check_file "app/src/main/AndroidManifest.xml"
check_file "app/src/main/cpp/CMakeLists.txt"
check_file "gradle/wrapper/gradle-wrapper.properties"

echo ""

# ────────────────────────────────────────────────
echo "━━━ 第 7 步：环境验证 ━━━"
echo ""
if [ -x "./gradlew" ] && [ -f "$JAR_PATH" ]; then
    SIZE=$(stat -c%s "$JAR_PATH" 2>/dev/null || stat -f%z "$JAR_PATH" 2>/dev/null)
    if [ "$SIZE" -gt 1000 ]; then
        echo "  尝试运行: ./gradlew --version"
        if ./gradlew --version >/dev/null 2>&1; then
            pass "Gradle wrapper 运行正常！"
        else
            warn "Gradle wrapper 运行失败（可能缺少依赖）"
        fi
    fi
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  安装检查完成                           ║"
echo "╠══════════════════════════════════════════╣"
echo "║  ✅ 通过: ${PASS}  ⚠️  警告: ${WARN}  ❌ 失败: ${FAIL}        ║"
echo "╚══════════════════════════════════════════╝"
echo ""

if [ "$FAIL" -eq 0 ]; then
    echo "🎉 环境就绪！编译命令："
    echo ""
    echo "   Debug APK:   ./build-apk.sh debug"
    echo "   Release APK: ./build-apk.sh release"
    echo ""
    echo "   或手动："
    echo "   ./gradlew assembleDebug"
    echo "   ./gradlew assembleRelease"
    echo ""
else
    echo "⚠️  有 ${FAIL} 项未通过，请按提示修复后重试"
    echo ""
fi
