package com.localaipainter.core

/**
 * UI 模块提供者 —— 插件式注册
 *
 * 新增 UI 模块的步骤：
 *   1. 创建对应的 Screen.kt 和 ViewModel.kt
 *   2. 在 PluginRegistry.registerBuiltinUIModules() 中调用 registerUIModule()
 *   3. 在 MainActivity 的 NavHost 中加一条路由
 *
 * 模块可声明依赖（如需要特定权限、特定后端），宿主会按需加载。
 */
data class UIModuleProvider(
    val id: String,
    val title: String,
    val icon: String,
    val route: String = id,
    val order: Int = 100,
    val requiresBackends: Set<String> = emptySet(),  // 需要的后端（空=通用）
    val requiresPermissions: Set<String> = emptySet(),
    val minSdk: Int = 26,
    val isExperimental: Boolean = false,
    val badge: String? = null,  // 角标文字，如 "NEW" "BETA"
    val enabledByDefault: Boolean = true,
    val dependencies: Set<String> = emptySet()  // 依赖的其他模块 id
) {
    fun isAvailable(currentSdk: Int): Boolean = currentSdk >= minSdk
}
