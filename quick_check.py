#!/usr/bin/env python3
"""Quick compile check - one file at a time, timeout per file."""
import subprocess, glob, os, signal

src_dir = 'app/src/main/cpp'
includes = ['-I' + src_dir, '-I' + src_dir + '/engine', '-I' + src_dir + '/v35']

cpp_files = sorted(glob.glob(src_dir + '/**/*.cpp', recursive=True))

def run_check(f):
    try:
        ret = subprocess.run(
            ['g++', '-std=c++17', '-fsyntax-only'] + includes + [f],
            capture_output=True, text=True, timeout=10
        )
        return ret.stdout + ret.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"

pass_c = warn_c = fail_c = timeout_c = 0
fails = []

for f in cpp_files:
    out = run_check(f).strip()
    rel = os.path.relpath(f)
    
    if not out:
        pass_c += 1
        print("  OK: " + rel)
    elif out == "TIMEOUT":
        timeout_c += 1
        print("  TIMEOUT: " + rel)
    elif 'No such file' in out and 'error:' not in out.replace('fatal error:', ''):
        warn_c += 1
        print("  WARN: " + rel)
    else:
        # Check if only no-such-file errors
        err_lines = [l for l in out.split('\n') if 'error:' in l]
        real = [l for l in err_lines if 'No such file' not in l]
        if not real:
            warn_c += 1
            print("  WARN: " + rel + " (NDK only)")
        else:
            fail_c += 1
            print("  FAIL: " + rel)
            for l in real[:3]:
                print("    " + l.strip()[:120])
            fails.append(f)

print("\n" + "="*50)
print("Pass: {}  Warn: {}  Fail: {}  Timeout: {}".format(pass_c, warn_c, fail_c, timeout_c))
print("Total: {}".format(len(cpp_files)))

if fails:
    print("\nFailed files:")
    for f in fails:
        print("  " + f)
