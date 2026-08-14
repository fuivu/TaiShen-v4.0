import subprocess
import os
import glob

ROOT = '/data/workspace/TaiShen-v4.0'
INC = os.path.join(ROOT, 'app/src/main/cpp')
STUB = os.path.join(ROOT, 'android_log_stub.h')

print("=" * 60)
print("FINAL COMPILATION VERIFICATION")
print("=" * 60)

# Copy stub to a location the compiler can find
# We'll use -include to force-include it
# Or better: create a fake android/ directory

fake_android = os.path.join(ROOT, 'fake_android')
os.makedirs(fake_android, exist_ok=True)
with open(os.path.join(fake_android, 'log.h'), 'w') as f:
    f.write('''#ifndef ANDROID_LOG_H
#define ANDROID_LOG_H
#include <cstdio>
#define ANDROID_LOG_VERBOSE 2
#define ANDROID_LOG_DEBUG   3
#define ANDROID_LOG_INFO    4
#define ANDROID_LOG_WARN    5
#define ANDROID_LOG_ERROR   6
#define ANDROID_LOG_FATAL   7
#define __android_log_print(prio, tag, ...) \\
    fprintf(stderr, "[%s] ", tag), fprintf(stderr, __VA_ARGS__), fprintf(stderr, "\\n")
#endif
''')

# Also create jni.h stub
with open(os.path.join(fake_android, 'jni.h'), 'w') as f:
    f.write('''#ifndef _JNI_H_
#define _JNI_H_
#include <stdint.h>
typedef int32_t jint;
typedef int64_t jlong;
typedef float jfloat;
typedef double jdouble;
typedef char16_t jchar;
typedef int8_t jbyte;
typedef int16_t jshort;
typedef int64_t jlong;
typedef int32_t jboolean;
typedef void* jobject;
typedef void* jclass;
typedef void* jstring;
typedef void* jarray;
typedef void* jintArray;
typedef void* jfloatArray;
typedef void* jbyteArray;
typedef void* JNIEnv;
typedef void* JavaVM;
typedef void* jmethodID;
typedef void* jfieldID;
struct _jobject {};
#endif
''')

# Key files
key_files = [
    'app/src/main/cpp/engine/lora/LoRAManager.cpp',
    'app/src/main/cpp/engine/controlnet/ControlNetProcessor.cpp',
    'app/src/main/cpp/engine/facerestore/FaceRestorer.cpp',
    'app/src/main/cpp/engine/mediatek/Dimensity8400Adapter.cpp',
]

print("\n--- Key files ---")
for kf in key_files:
    f = os.path.join(ROOT, kf)
    if not os.path.exists(f):
        print(f"  SKIP: {kf}")
        continue
    r = subprocess.run(
        ['g++', '-std=c++17', '-fsyntax-only',
         '-I', INC, '-I', fake_android,
         '-DANDROID', '-DJNIEXPORT=', '-DJNICALL=', f],
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
print("\n--- All C++ files ---")
cpp_files = glob.glob(os.path.join(ROOT, '**/*.cpp'), recursive=True)
failed = []
for cf in sorted(cpp_files):
    r = subprocess.run(
        ['g++', '-std=c++17', '-fsyntax-only',
         '-I', INC, '-I', fake_android,
         '-DANDROID', '-DJNIEXPORT=', '-DJNICALL=', cf],
        capture_output=True, text=True, timeout=30
    )
    if r.returncode != 0:
        rel = os.path.relpath(cf, ROOT)
        err_lines = [l for l in r.stderr.strip().split('\n') if l.strip()]
        err = err_lines[-1] if err_lines else "unknown"
        failed.append((rel, err))

if failed:
    print(f"  {len(failed)} FAILED:")
    for rel, err in failed[:25]:
        print(f"    {rel}: {err[:120]}")
else:
    print(f"  PASS: All {len(cpp_files)} C++ files compile cleanly")

# Now check headers too
print("\n--- All C++ headers ---")
h_files = glob.glob(os.path.join(ROOT, '**/*.h'), recursive=True)
failed_h = []
for hf in sorted(h_files):
    r = subprocess.run(
        ['g++', '-std=c++17', '-fsyntax-only',
         '-I', INC, '-I', fake_android,
         '-DANDROID', '-DJNIEXPORT=', '-DJNICALL=', hf],
        capture_output=True, text=True, timeout=30
    )
    if r.returncode != 0:
        rel = os.path.relpath(hf, ROOT)
        err_lines = [l for l in r.stderr.strip().split('\n') if l.strip()]
        err = err_lines[-1] if err_lines else "unknown"
        failed_h.append((rel, err))

if failed_h:
    print(f"  {len(failed_h)} FAILED:")
    for rel, err in failed_h[:15]:
        print(f"    {rel}: {err[:120]}")
else:
    print(f"  PASS: All {len(h_files)} header files compile cleanly")

print("\n" + "=" * 60)
total_fail = len(failed) + len(failed_h)
total_files = len(cpp_files) + len(h_files)
print(f"TOTAL: {total_files - total_fail}/{total_files} files PASS")
if total_fail == 0:
    print("RESULT: ALL FILES COMPILE CLEANLY")
else:
    print(f"RESULT: {total_fail} files need attention")
print("=" * 60)
