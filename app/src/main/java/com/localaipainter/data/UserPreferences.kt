package com.localaipainter.data

import android.content.Context
import androidx.datastore.preferences.core.*
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "user_preferences")

/**
 * 用户偏好设置 — DataStore 封装
 * 14 项设置
 */
class UserPreferences(private val context: Context) {

    private val dataStore = context.dataStore

    // ============ Keys ============

    companion object {
        val KEY_THEME              = stringPreferencesKey("theme")
        val KEY_POWER_MODE         = stringPreferencesKey("power_mode")
        val KEY_DEFAULT_STEPS     = intPreferencesKey("default_steps")
        val KEY_DEFAULT_CFG       = floatPreferencesKey("default_cfg")
        val KEY_DEFAULT_WIDTH     = intPreferencesKey("default_width")
        val KEY_DEFAULT_HEIGHT    = intPreferencesKey("default_height")
        val KEY_DEFAULT_SCHEDULER = stringPreferencesKey("default_scheduler")
        val KEY_GPU_CACHE         = booleanPreferencesKey("gpu_cache")
        val KEY_AUTO_START        = booleanPreferencesKey("auto_start")
        val KEY_KEEP_SCREEN_ON   = booleanPreferencesKey("keep_screen_on")
        val KEY_LAST_MODEL_PATH   = stringPreferencesKey("last_model_path")
        val KEY_LAST_VAE_PATH     = stringPreferencesKey("last_vae_path")
        val KEY_THREADS           = intPreferencesKey("threads")
        val KEY_USE_GPU          = booleanPreferencesKey("use_gpu")
        val KEY_DEFAULT_LORA_DIR  = stringPreferencesKey("default_lora_dir")
    }

    // ============ Flow 读取 ============

    val theme: Flow<String> = dataStore.data.map { it[KEY_THEME] ?: "dark" }
    val powerMode: Flow<String> = dataStore.data.map { it[KEY_POWER_MODE] ?: "balanced" }
    val defaultSteps: Flow<Int> = dataStore.data.map { it[KEY_DEFAULT_STEPS] ?: 30 }
    val defaultCFG: Flow<Float> = dataStore.data.map { it[KEY_DEFAULT_CFG] ?: 7.5f }
    val defaultWidth: Flow<Int> = dataStore.data.map { it[KEY_DEFAULT_WIDTH] ?: 512 }
    val defaultHeight: Flow<Int> = dataStore.data.map { it[KEY_DEFAULT_HEIGHT] ?: 512 }
    val defaultScheduler: Flow<String> = dataStore.data.map { it[KEY_DEFAULT_SCHEDULER] ?: "euler_a" }
    val gpuCacheEnabled: Flow<Boolean> = dataStore.data.map { it[KEY_GPU_CACHE] ?: true }
    val autoStartEnabled: Flow<Boolean> = dataStore.data.map { it[KEY_AUTO_START] ?: false }
    val keepScreenOnEnabled: Flow<Boolean> = dataStore.data.map { it[KEY_KEEP_SCREEN_ON] ?: true }
    val lastModelPath: Flow<String> = dataStore.data.map { it[KEY_LAST_MODEL_PATH] ?: "" }
    val threads: Flow<Int> = dataStore.data.map { it[KEY_THREADS] ?: 4 }
    val useGpu: Flow<Boolean> = dataStore.data.map { it[KEY_USE_GPU] ?: true }

    // ============ 写入 ============

    suspend fun setTheme(theme: String) = dataStore.edit { it[KEY_THEME] = theme }
    suspend fun setPowerMode(mode: String) = dataStore.edit { it[KEY_POWER_MODE] = mode }
    suspend fun setDefaultSteps(steps: Int) = dataStore.edit { it[KEY_DEFAULT_STEPS] = steps }
    suspend fun setDefaultCFG(cfg: Float) = dataStore.edit { it[KEY_DEFAULT_CFG] = cfg }
    suspend fun setDefaultResolution(w: Int, h: Int) = dataStore.edit { prefs ->
        prefs[KEY_DEFAULT_WIDTH] = w
        prefs[KEY_DEFAULT_HEIGHT] = h
    }
    suspend fun setDefaultScheduler(scheduler: String) = dataStore.edit { it[KEY_DEFAULT_SCHEDULER] = scheduler }
    suspend fun setGpuCacheEnabled(enabled: Boolean) = dataStore.edit { it[KEY_GPU_CACHE] = enabled }
    suspend fun setAutoStart(enabled: Boolean) = dataStore.edit { it[KEY_AUTO_START] = enabled }
    suspend fun setKeepScreenOn(enabled: Boolean) = dataStore.edit { it[KEY_KEEP_SCREEN_ON] = enabled }
    suspend fun setLastModelPath(path: String) = dataStore.edit { it[KEY_LAST_MODEL_PATH] = path }
    suspend fun setThreads(threads: Int) = dataStore.edit { it[KEY_THREADS] = threads }
    suspend fun setUseGpu(use: Boolean) = dataStore.edit { it[KEY_USE_GPU] = use }

    // ============ 批量导出/导入 ============

    suspend fun exportAll(): Map<String, Any> {
        val prefs = dataStore.data.map { it.toPreferences() }.kotlinx_coroutines_flow_first()
        return prefs.asMap().mapKeys { it.key.name }
    }

    suspend fun importAll(map: Map<String, Any>) {
        dataStore.edit { prefs ->
            map.forEach { (k, v) ->
                when (v) {
                    is String  -> prefs[stringPreferencesKey(k)] = v
                    is Int     -> prefs[intPreferencesKey(k)] = v
                    is Float   -> prefs[floatPreferencesKey(k)] = v
                    is Boolean -> prefs[booleanPreferencesKey(k)] = v
                    is Long    -> prefs[longPreferencesKey(k)] = v
                }
            }
        }
    }
}
