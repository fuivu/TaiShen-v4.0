// ═════════════════════════════════════════════════════
//  TaiShen Architecture v4.0 — App Build File (升级版)
//  Local AI Painter — 太神架构 4.0
//  深度融合：6推理后端 + 14调度器 + 6 ControlNet
//            + 5路LoRA + 超分/人脸修复 + 天玑NPU
//            + Vulkan零拷贝 + INT2/FP8量化 + 图融合
// ═════════════════════════════════════════════════════

import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.room)
    alias(libs.plugins.ksp)
}

android {
    namespace = "com.localaipainter"
    compileSdk = libs.versions.compileSdk.get().toInt()

    defaultConfig {
        applicationId = "com.localaipainter"
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.targetSdk.get().toInt()
        versionCode = 40
        versionName = "4.0.0"

        // ─── ABI 架构 ────────────────────────────────
        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        // ─── MultiDex（方法数超限保护）──────────────
        multiDexEnabled = true

        // ─── Vector 兼容 ──────────────────────────────
        vectorDrawables {
            useSupportLibrary = true
        }

        // ─── C++ 编译参数 ────────────────────────────
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17 -O3 -flto -fopenmp"
                arguments += "-DANDROID_STL=c++_shared"
                arguments += "-DANDROID_PLATFORM=android-26"
            }
        }

        // ⚠️ ldLibs 已弃用 (AGP 7.0+)
        // 实际链接在 CMakeLists.txt 中通过 target_link_libraries 完成
        // 此处保留仅为兼容旧 NDK 行为，后续可安全移除
        @Suppress("DEPRECATION")
        ndk {
            ldLibs?.addAll(listOf("vulkan", "OpenCL", "neuralnetworks"))
        }

        // ─── NDK 版本锁定 ────────────────────────────
        ndkVersion = libs.versions.ndk.get()
    }

    // ═══════════════════════════════════════════════
    //  Room Schema 配置
    //  位置：android 块内、defaultConfig 外部 ✅
    // ═══════════════════════════════════════════════
    room {
        schemaDirectory("$projectDir/schemas")
    }

    // ═══════════════════════════════════════════════
    //  构建类型
    // ═══════════════════════════════════════════════
    buildTypes {
        debug {
            isMinifyEnabled = false
            isShrinkResources = false
            applicationIdSuffix = ".debug"
            // Debug 不混淆，加速编译
        }

        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )

            // ─── 不压缩的模型文件格式 ────────────────
            androidResources {
                noCompress += listOf(
                    "model", "safetensors", "gguf",
                    "onnx", "mnn", "ncnn", "tflite"
                )
            }
        }
    }

    // ═══════════════════════════════════════════════
    //  构建特性
    // ═══════════════════════════════════════════════
    buildFeatures {
        compose = true
        aidl = false
        buildConfig = true
    }

    // ✅ Kotlin 2.1.0 ↔ Compose Compiler 1.5.15 精确对齐
    composeOptions {
        kotlinCompilerExtensionVersion = libs.versions.composeCompiler.get()
    }

    // ─── Java 17 ────────────────────────────────────
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_17)
            // 严格模式：所有警告视为错误（仅 release）
            // allWarningsAsErrors.set(true)
        }
    }

    // ─── 资源排除规则 ────────────────────────────────
    packaging {
        resources {
            excludes += listOf(
                "/META-INF/{AL2.0,LGPL2.1}",
                "/META-INF/DEPENDENCIES",
                "/META-INF/LICENSE*",
                "/META-INF/NOTICE*",
                "/META-INF/INDEX.LIST"
            )
        }
        jniLibs {
            useLegacyPackaging = false
        }
    }

    // ─── CMake 配置 ──────────────────────────────────
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    // ─── 源码集 ──────────────────────────────────────
    sourceSets {
        getByName("main") {
            java.srcDirs("src/main/java")
            res.srcDirs("src/main/res")
            manifest.srcFile("src/main/AndroidManifest.xml")
        }
        getByName("test") {
            java.srcDirs("src/test/java")
        }
        getByName("androidTest") {
            java.srcDirs("src/androidTest/java")
        }
    }

    // ─── Lint 配置 ───────────────────────────────────
    lint {
        abortOnError = false
        checkReleaseBuilds = false
        disable += listOf("MissingTranslation", "ExtraTranslation")
    }
}

// ═════════════════════════════════════════════════════
//  KSP 配置（Room 注解处理）
// ═════════════════════════════════════════════════════
ksp {
    arg("room.schemaLocation", "$projectDir/schemas")
    arg("room.incremental", "true")
    arg("room.expandProjection", "true")
}

// ═════════════════════════════════════════════════════
//  依赖声明（按模块分组，清晰可维护）
// ═════════════════════════════════════════════════════
dependencies {

    // ─── Android Core ────────────────────────────────
    implementation(libs.androidx.core)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.splashscreen)

    // ─── Compose BOM（物料清单，统一管理版本）────────
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.material.icons)
    implementation(libs.androidx.compose.foundation)
    implementation(libs.androidx.compose.animation)

    // ─── Activity & Navigation ──────────────────────
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.navigation.compose)

    // ─── ViewModel & Lifecycle ──────────────────────
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.ktx)
    implementation(libs.androidx.lifecycle.service)

    // ─── Coroutines ──────────────────────────────────
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.coroutines.core)

    // ─── Serialization ───────────────────────────────
    implementation(libs.kotlinx.serialization.json)

    // ─── Room（数据库）───────────────────────────────
    implementation(libs.androidx.room.runtime)
    implementation(libs.androidx.room.ktx)
    ksp(libs.androidx.room.compiler)

    // ─── DataStore（轻量 KV 存储）────────────────────
    implementation(libs.androidx.datastore.preferences)

    // ─── Coil 图片加载（正确别名：coil-compose）─────
    implementation(libs.coil.compose)
    implementation(libs.coil.gif)

    // ─── Biometric 生物识别 ─────────────────────────
    implementation(libs.androidx.biometric)

    // ─── ONNX Runtime（推理后端之一）────────────────
    implementation(libs.onnx.runtime.android)

    // ─── GPUImage 滤镜/后处理 ───────────────────────
    implementation(libs.gpuimage)

    // ─── WorkManager 后台任务 ────────────────────────
    implementation(libs.workmanager.ktx)

    // ─── Ktor 网络（更新检查/模型下载）──────────────
    implementation(libs.ktor.client.core)
    implementation(libs.ktor.client.okhttp)
    implementation(libs.ktor.client.content.negotiation)
    implementation(libs.ktor.serialization.json)

    // ─── OkHttp（备用网络层）─────────────────────────
    implementation(libs.okhttp)

    // ─── 调试工具 ─────────────────────────────────────
    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.compose.test.manifest)
    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.compose.test.junit)
}
