// ══════════════════════════════════════════════════════
//  TaiShen Architecture v4.0 — Root Build File
//  Local AI Painter — 太神架构
//  全局插件声明 + 统一版本治理
// ══════════════════════════════════════════════════════

plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.compose) apply false
    alias(libs.plugins.kotlin.serialization) apply false
    alias(libs.plugins.room) apply false
    alias(libs.plugins.ksp) apply false
}

// ─── 全局构建配置 ─────────────────────────────────────
allprojects {
    // 统一 JVM 目标
    tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }

    // 统一编码
    tasks.withType<JavaCompile> {
        options.encoding = "UTF-8"
    }
}

// ─── 清理任务 ─────────────────────────────────────────
tasks.register("clean", Delete::class) {
    delete(rootProject.buildDir)
}
