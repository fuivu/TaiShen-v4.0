// ═══════════════════════════════════════════════
//  TaiShen Architecture v4.0 — Settings
//  Local AI Painter — 太神架构
// ═══════════════════════════════════════════════

pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // ─── JitPack（GPUImage 等第三方库源）──────
        maven { url = uri("https://jitpack.io") }
    }
}

// ─── 项目命名 ───────────────────────────────────
rootProject.name = "LocalAIPainter"
include(":app")
