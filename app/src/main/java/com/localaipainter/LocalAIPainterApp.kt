package com.localaipainter

import android.app.Application
import android.util.Log
import com.localaipainter.core.PluginRegistry
import com.localaipainter.data.AppDatabase
import com.localaipainter.data.UserPreferences
import com.localaipainter.data.dao.ModelDao
import com.localaipainter.engine.HeterogeneousPipeline
import com.localaipainter.memory.LruTensorCache
import com.localaipainter.util.CrashHandler
import com.localaipainter.util.Logger

class LocalAIPainterApp : Application() {

    lateinit var userPreferences: UserPreferences
        private set

    lateinit var heteroPipeline: HeterogeneousPipeline
        private set

    lateinit var tensorCache: LruTensorCache
        private set

    /** 数据库 DAO（全局可访问） */
    val modelDao: ModelDao by lazy {
        AppDatabase.getInstance(this).modelDao()
    }

    /** 模型仓库单例 */
    val modelRepository: com.localaipainter.data.repository.ModelRepository by lazy {
        com.localaipainter.data.repository.ModelRepository(
            AppDatabase.getInstance(this).modelDao()
        )
    }

    override fun onCreate() {
        super.onCreate()
        instance = this

        // 初始化日志和崩溃捕获
        Logger.init(this)
        CrashHandler.init(this)

        Logger.i(TAG, "=== Local AI Painter v4.0 (TaiShen Architecture) starting ===")

        // 初始化用户偏好
        userPreferences = UserPreferences(this)

        // 初始化插件系统
        PluginRegistry.initAll(this)

        // 初始化异构推理管线
        tensorCache = LruTensorCache(maxSizeMB = 1024)
        heteroPipeline = HeterogeneousPipeline(this, tensorCache)
        heteroPipeline.init()

        // 初始化 EngineFactory（供 PluginRegistry 使用）
        com.localaipainter.engine.EngineFactory.init(this)

        // 创建必要的目录
        getExternalFilesDir("models")?.mkdirs()
        getExternalFilesDir("models/lora")?.mkdirs()
        getExternalFilesDir("models/controlnet")?.mkdirs()
        getExternalFilesDir("outputs")?.mkdirs()
        getExternalFilesDir("outputs/upscaled")?.mkdirs()
        getExternalFilesDir("logs")?.mkdirs()

        Logger.i(TAG, "App init complete")
    }

    override fun onTerminate() {
        heteroPipeline.release()
        tensorCache.clear()
        PluginRegistry.shutdown()
        Logger.i(TAG, "App terminated, resources released")
        super.onTerminate()
    }

    companion object {
        private const val TAG = "LocalAIPainter"
        lateinit var instance: LocalAIPainterApp
            private set
    }
}
