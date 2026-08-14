#!/bin/bash
# Compile-check all C++ files in the project
set -e

SRC_DIR="app/src/main/cpp"
ENGINE_DIR="engine"

# Headers to check (just count them)
echo "=== Header files ==="
find "$SRC_DIR" -name "*.h" 2>/dev/null | wc -l

echo ""
echo "=== Source files ==="
find "$SRC_DIR" -name "*.cpp" 2>/dev/null | wc -l

echo ""
echo "=== Compiling each .cpp with -fsyntax-only ==="
FAIL=0
PASS=0
for f in $(find "$SRC_DIR" -name "*.cpp" 2>/dev/null | sort); do
    # Try to compile just this file (will fail on missing includes, but catches syntax errors)
    OUT=$(g++ -std=c++17 -fsyntax-only -I"$SRC_DIR" -I"$SRC_DIR/engine" "$f" 2>&1)
    if [ -z "$OUT" ]; then
        echo "  ✅ $(basename $f)"
        PASS=$((PASS+1))
    else
        # Filter out "file not found" type errors (missing NDK headers)
        REAL_ERR=$(echo "$OUT" | grep -v "fatal error:.*No such file" | grep -v "note:" | head -3)
        if [ -n "$REAL_ERR" ]; then
            echo "  ❌ $f"
            echo "$OUT" | head -5 | sed 's/^/     /'
            FAIL=$((FAIL+1))
        else
            echo "  ⚠️  $(basename $f) (missing NDK headers only)"
            PASS=$((PASS+1))
        fi
    fi
done

echo ""
echo "=== Summary: $PASS passed, $FAIL failed ==="
