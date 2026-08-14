package com.localaipainter.lora

import android.util.Log
import com.localaipainter.engine.InferenceEngine
import com.localaipainter.data.entity.ModelEntity
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * LoRA 管理器 - 加载、堆叠、缩放、开关
 */
class LoraManager(private val engine: InferenceEngine) {

    private val loadedLoras = mutableMapOf<String, LoraWeight>()
    private val mutex = Mutex()

    /**
     * 加载 LoRA 权重
     */
    suspend fun load(lora: LoraWeight): Result<Unit> = mutex.withLock {
        return try {
            if (loadedLoras.containsKey(lora.name)) {
                return@withLock Result.success(Unit)
            }
            val ret = nativeLoadLora(lora.path, lora.scale, lora.rank, lora.alpha)
            if (ret != 0) return@withLock Result.failure(Exception("LoRA load failed: $ret"))
            loadedLoras[lora.name] = lora
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * 卸载 LoRA
     */
    suspend fun unload(name: String): Result<Unit> = mutex.withLock {
        return try {
            val lora = loadedLoras[name] ?: return@withLock Result.success(Unit)
            val ret = nativeUnloadLora(lora.path)
            if (ret != 0) return@withLock Result.failure(Exception("LoRA unload failed: $ret"))
            loadedLoras.remove(name)
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * 设置 LoRA 缩放权重
     */
    suspend fun setScale(name: String, scale: Float): Result<Unit> = mutex.withLock {
        val lora = loadedLoras[name] ?: return Result.failure(IllegalArgumentException("LoRA not loaded: $name"))
        return try {
            val ret = nativeSetLoraScale(lora.path, scale)
            if (ret != 0) return Result.failure(Exception("Set scale failed: $ret"))
            loadedLoras[name] = lora.copy(scale = scale)
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * 获取当前加载的所有 LoRA
     */
    fun getLoadedLoras(): List<LoraWeight> = loadedLoras.values.toList()

    /**
     * 应用所有 LoRA 到当前模型
     */
    suspend fun applyAll(): Result<Unit> = mutex.withLock {
        return try {
            val loras = loadedLoras.values.toList()
            val ret = nativeApplyLoras(loras.toTypedArray())
            if (ret != 0) return Result.failure(Exception("Apply Loras failed: $ret"))
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * 清除所有 LoRA
     */
    suspend fun clearAll(): Result<Unit> = mutex.withLock {
        return try {
            val ret = nativeClearLoras()
            if (ret != 0) return Result.failure(Exception("Clear Loras failed: $ret"))
            loadedLoras.clear()
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    fun release() {
        loadedLoras.clear()
        nativeRelease()
    }

    private external fun nativeLoadLora(path: String, scale: Float, rank: Int, alpha: Float): Int
    private external fun nativeUnloadLora(path: String): Int
    private external fun nativeSetLoraScale(path: String, scale: Float): Int
    private external fun nativeApplyLoras(loras: Array<LoraWeight>): Int
    private external fun nativeClearLoras(): Int
    private external fun nativeRelease(): Int

    companion object {
        private const val TAG = "LoraManager"
    }
}
