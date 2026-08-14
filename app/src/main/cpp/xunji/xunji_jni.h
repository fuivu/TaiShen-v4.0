/*
 * ═════════════════════════════════════════════════════════════
 *  迅集 (XunJi) — JNI 桥接  v4.0 TaiShen
 *  15 个 native 方法 · 类型安全 · 异常保护
 * ═════════════════════════════════════════════════════════════
 */
#pragma once
#include <jni.h>
#include "../xunji/xunji_quantizer.h"
#include "../xunji/xunji_memory.h"

extern "C" {

// ─── 量化 ─────────────────────────────────────────────────
JNIEXPORT jint JNICALL Java_com_localaipainter_xunji_XunJiQuantExecutor_nativeSetPrecision
  (JNIEnv*, jobject, jstring level);
JNIEXPORT jlong JNICALL Java_com_localaipainter_xunji_XunJiQuantExecutor_nativeLoadModel
  (JNIEnv*, jobject, jstring path, jstring level);
JNIEXPORT void JNICALL Java_com_localaipainter_xunji_XunJiQuantExecutor_nativeFreeHandle
  (JNIEnv*, jobject, jlong handle);
JNIEXPORT jfloatArray JNICALL Java_com_localaipainter_xunji_XunJiQuantExecutor_nativeRun
  (JNIEnv*, jobject, jlong handle, jfloatArray input);

// ─── 自动选择 ────────────────────────────────────────────
JNIEXPORT jstring JNICALL Java_com_localaipainter_xunji_XunJiScheduler_nativeAutoSelect
  (JNIEnv*, jobject, jlong totalKb, jlong availKb, jlong gpuTotalKb, jlong gpuAvailKb);

// ─── 内存 ────────────────────────────────────────────────
JNIEXPORT jlong JNICALL Java_com_localaipainter_xunji_XunJiMemoryMonitor_nativeSample
  (JNIEnv*, jobject);
JNIEXPORT jint JNICALL Java_com_localaipainter_xunji_XunJiMemoryMonitor_nativeGetPressure
  (JNIEnv*, jobject);
JNIEXPORT jlong JNICALL Java_com_localaipainter_xunji_XunJiMemoryMonitor_nativeAvailKb
  (JNIEnv*, jobject);
JNIEXPORT void JNICALL Java_com_localaipainter_xunji_XunJiMemoryMonitor_nativeStartMonitor
  (JNIEnv*, jobject);
JNIEXPORT void JNICALL Java_com_localaipainter_xunji_XunJiMemoryMonitor_nativeStopMonitor
  (JNIEnv*, jobject);

// ─── 工具 ────────────────────────────────────────────────
JNIEXPORT jint JNICALL Java_com_localaipainter_xunji_XunJiQuantExecutor_nativeGetBits
  (JNIEnv*, jobject, jstring level);
JNIEXPORT jfloat JNICALL Java_com_localaipainter_xunji_XunJiQuantExecutor_nativeGetMemRatio
  (JNIEnv*, jobject, jstring level);
JNIEXPORT void JNICALL Java_com_localaipainter_xunji_XunJiMemoryMonitor_nativeReportLeaks
  (JNIEnv*, jobject);

} // extern "C"
