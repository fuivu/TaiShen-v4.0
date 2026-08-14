package com.localaipainter.ui.settings

import android.app.Application
import androidx.lifecycle.*
import com.localaipainter.data.preferences.UserPreferences
import com.localaipainter.engine.DeviceInfo
import com.localaipainter.engine.EngineFactory
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

class SettingsViewModel(
    application: Application,
    private val prefs: UserPreferences,
    private val engine: EngineFactory
) : AndroidViewModel(application) {

    // ============ Theme ============
    val theme: StateFlow<String> = prefs.theme.stateIn(
        viewModelScope, SharingStarted.Eagerly, "dark")

    fun setTheme(theme: String) {
        viewModelScope.launch { prefs.setTheme(theme) }
    }

    // ============ Performance ============
    val useGpu: StateFlow<Boolean> = prefs.useGpu.stateIn(
        viewModelScope, SharingStarted.Eagerly, true)
    val useQuantization: StateFlow<Boolean> = prefs.useQuantization.stateIn(
        viewModelScope, SharingStarted.Eagerly, true)
    val threadCount: StateFlow<Int> = prefs.threadCount.stateIn(
        viewModelScope, SharingStarted.Eagerly, 4)
    val memoryLimitMB: StateFlow<Int> = prefs.memoryLimitMB.stateIn(
        viewModelScope, SharingStarted.Eagerly, 4096)

    fun setUseGpu(enabled: Boolean) {
        viewModelScope.launch { prefs.setUseGpu(enabled) }
    }
    fun setUseQuantization(enabled: Boolean) {
        viewModelScope.launch { prefs.setUseQuantization(enabled) }
    }
    fun setThreadCount(count: Int) {
        viewModelScope.launch { prefs.setThreadCount(count) }
    }
    fun setMemoryLimitMB(mb: Int) {
        viewModelScope.launch { prefs.setMemoryLimitMB(mb) }
    }

    // ============ Defaults ============
    val defaultSteps: StateFlow<Int> = prefs.defaultSteps.stateIn(
        viewModelScope, SharingStarted.Eagerly, 20)
    val defaultCfg: StateFlow<Float> = prefs.defaultCfg.stateIn(
        viewModelScope, SharingStarted.Eagerly, 7.5f)

    fun setDefaultSteps(steps: Int) {
        viewModelScope.launch { prefs.setDefaultSteps(steps) }
    }
    fun setDefaultCfg(cfg: Float) {
        viewModelScope.launch { prefs.setDefaultCfg(cfg) }
    }

    // ============ System ============
    val autoStart: StateFlow<Boolean> = prefs.autoStart.stateIn(
        viewModelScope, SharingStarted.Eagerly, false)
    val keepScreenOn: StateFlow<Boolean> = prefs.keepScreenOn.stateIn(
        viewModelScope, SharingStarted.Eagerly, true)
    val saveMetadata: StateFlow<Boolean> = prefs.saveMetadata.stateIn(
        viewModelScope, SharingStarted.Eagerly, true)
    val outputFormat: StateFlow<String> = prefs.outputFormat.stateIn(
        viewModelScope, SharingStarted.Eagerly, "PNG")

    fun setAutoStart(enabled: Boolean) {
        viewModelScope.launch { prefs.setAutoStart(enabled) }
    }
    fun setKeepScreenOn(enabled: Boolean) {
        viewModelScope.launch { prefs.setKeepScreenOn(enabled) }
    }
    fun setSaveMetadata(enabled: Boolean) {
        viewModelScope.launch { prefs.setSaveMetadata(enabled) }
    }
    fun setOutputFormat(format: String) {
        viewModelScope.launch { prefs.setOutputFormat(format) }
    }

    // ============ Device Info ============
    private val _deviceInfo = MutableStateFlow<DeviceInfo?>(null)
    val deviceInfo: StateFlow<DeviceInfo?> = _deviceInfo

    fun refreshDeviceInfo() {
        val info = engine.getDeviceInfo()
        _deviceInfo.value = info
        // 自动应用优化
        engine.applyDeviceOptimizations()
    }

    // ============ Export Log ============
    fun exportLog(): String {
        val sb = StringBuilder()
        sb.appendLine("=== Local AI Painter v4.0 (TaiShen) Debug Log ===")
        sb.appendLine("Time: ${System.currentTimeMillis()}")
        _deviceInfo.value?.let { d ->
            sb.appendLine("Device: ${d.cpuModel}")
            sb.appendLine("Android: ${d.androidVersion} (SDK ${d.sdkVersion})")
            sb.appendLine("RAM: ${d.totalRamMB}MB total, ${d.availableRamMB}MB avail")
            sb.appendLine("GPU: ${if (d.hasGpu) "Yes (${d.gpuMemoryMB}MB)" else "No"}")
            sb.appendLine("Best Backend: ${d.bestBackend}")
            sb.appendLine("Recommended Max Res: ${d.recommendedMaxResolution}")
        }
        sb.appendLine("Engine ID: ${engine.engineId}")
        sb.appendLine("=== End Log ===")
        return sb.toString()
    }
}
