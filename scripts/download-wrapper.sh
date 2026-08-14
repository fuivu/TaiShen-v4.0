#!/bin/bash
# Downloads the correct gradle-wrapper.jar for Gradle 8.7
# Run this once on a machine with internet access, then commit the jar.
set -e
WRAPPER_URL="https://github.com/gradle/gradle/raw/v8.7.0/gradle/wrapper/gradle-wrapper.jar"
DEST="gradle/wrapper/gradle-wrapper.jar"
mkdir -p "$(dirname "$DEST")"
echo "Downloading gradle-wrapper.jar..."
curl -L --fail "$WRAPPER_URL" -o "$DEST"
echo "Done! File size: $(wc -c < "$DEST") bytes"
echo "If curl fails, manually download from:"
echo "  $WRAPPER_URL"
echo "And place it at: $DEST"
