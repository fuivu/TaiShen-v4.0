# ═════════════════════════════════════════════════════
#  TaiShen Architecture v4.0 — ProGuard Rules
#  Local AI Painter — 太神架构
# ═════════════════════════════════════════════════════

# ─── 基础保留 ──────────────────────────────────────
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile
-keepattributes Exceptions,InnerClasses,Signature,Deprecated,Annotation
-keepattributes *Annotation*
-keepattributes EnclosingMethod
-keepattributes RuntimeVisibleAnnotations,RuntimeVisibleParameterAnnotations

# ─── Kotlin 元数据 ──────────────────────────────────
-dontwarn kotlinx.coroutines.**
-keep class kotlinx.coroutines.** { *; }
-keep class kotlin.Metadata { *; }
-keepclassmembers class **$*Coroutine$* {
    volatile <fields>;
}

# ─── Kotlinx Serialization ──────────────────────────
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.**
-keep class kotlinx.serialization.** { *; }
-keepclassmembers class **$Companion {
    @kotlinx.serialization.Serializable *;
}
-keepclasseswithmembers class * {
    @kotlinx.serialization.Serializable <fields>;
}

# ─── JNI / Native 方法（必须保留签名）──────────────
-keepclasseswithmembernames class * {
    native <methods>;
}
-keep class com.localaipainter.**.jni.** { *; }
-keep class com.localaipainter.engine.JNIBridge { *; }
-keep class com.localaipainter.engine.v35.EngineV35 { *; }

# ─── Room 数据库 ────────────────────────────────────
-keep class androidx.room.** { *; }
-keep @androidx.room.Entity class *
-dontwarn androidx.room.paging.**

# ─── ONNX Runtime ───────────────────────────────────
-keep class com.microsoft.onnxruntime.** { *; }
-dontwarn com.microsoft.onnxruntime.**

# ─── Coil 图片加载 ──────────────────────────────────
-dontwarn coil.**
-keep class coil.** { *; }

# ─── GPUImage ────────────────────────────────────────
-keep class jp.co.cyberagent.android.gpuimage.** { *; }
-dontwarn jp.co.cyberagent.android.gpuimage.**

# ─── WorkManager ────────────────────────────────────
-keep class androidx.work.** { *; }
-dontwarn androidx.work.**

# ─── Ktor 网络 ──────────────────────────────────────
-dontwarn io.ktor.**
-keep class io.ktor.** { *; }

# ─── OkHttp ─────────────────────────────────────────
-dontwarn okhttp3.**
-dontwarn okio.**
-keep class okhttp3.** { *; }
-keep class okio.** { *; }

# ─── Biometric ──────────────────────────────────────
-keep class androidx.biometric.** { *; }

# ─── Compose ────────────────────────────────────────
-dontwarn androidx.compose.**
-keep class androidx.compose.** { *; }

# ─── DataStore ──────────────────────────────────────
-keep class androidx.datastore.** { *; }

# ─── 模型文件不混淆 ─────────────────────────────────
-keep class com.localaipainter.models.** { *; }
-keep class com.localaipainter.data.entity.** { *; }
-keep class com.localaipainter.data.dao.** { *; }

# ─── 推理引擎类 ────────────────────────────────────
-keep class com.localaipainter.engine.** { *; }
-keep class com.localaipainter.pipeline.** { *; }
-keep class com.localaipainter.quantize.** { *; }
-keep class com.localaipainter.lora.** { *; }

# ─── Application 入口 ──────────────────────────────
-keep class com.localaipainter.LocalAIPainterApp { *; }
-keep class com.localaipainter.App { *; }

# ─── 反射调用保留 ──────────────────────────────────
-keep class com.localaipainter.core.PluginRegistry { *; }
-keep class com.localaipainter.core.FeatureToggle { *; }

# ─── 移除日志（Release 优化）──────────────────────
-assumenosideeffects class android.util.Log {
    public static *** d(...);
    public static *** v(...);
}
-assumenosideeffects class com.localaipainter.util.Logger {
    public static *** d(...);
    public static *** v(...);
}

# ─── 通用 ───────────────────────────────────────────
-optimizationpasses 5
-allowaccessmodification
-repackageclasses ''
-verbose
