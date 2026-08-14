package com.localaipainter.task

import android.content.Context
import com.localaipainter.core.FeatureToggle
import com.localaipainter.data.entity.TaskEntity
import com.localaipainter.data.repository.TaskRepository
import com.localaipainter.util.Logger
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

/**
 * 任务队列管理器 —— 支持暂停/恢复/取消/重试/优先级
 *
 * 扩展点（未来新增任务类型只需）：
 *   1. 实现 TaskHandler 接口
 *   2. 在 TaskType 枚举中加一项
 *   3. 在 executeTask() 的 when 中加一行分发
 */
class TaskQueueManager(
    private val context: Context,
    private val taskRepo: TaskRepository
) {
    companion object { private const val TAG = "TaskQueue" }

    enum class TaskType {
        TEXT_TO_IMAGE,    // 文生图
        IMAGE_TO_IMAGE,  // 图生图
        INPAINT,         // 局部重绘
        UPSCALE,         // 超分
        MODEL_DOWNLOAD,   // 模型下载
        BATCH_EXPORT      // 批量导出
    }

    enum class QueueState { IDLE, RUNNING, PAUSED, ERROR }

    // ---- 状态流 ----
    private val _state = MutableStateFlow(QueueState.IDLE)
    val state: StateFlow<QueueState> = _state.asStateFlow()

    private val _progress = MutableStateFlow(0f)
    val progress: StateFlow<Float> = _progress.asStateFlow()

    private val _currentTask = MutableStateFlow<TaskEntity?>(null)
    val currentTask: StateFlow<TaskEntity?> = _currentTask.asStateFlow()

    private val _taskLog = MutableSharedFlow<TaskLog>(extraBufferCapacity = 64)
    val taskLog: SharedFlow<TaskLog> = _taskLog.asSharedFlow()

    // ---- 内部 ----
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var isPaused = false
    private var currentJob: Job? = null
    private val handlerRegistry = linkedMapOf<TaskType, TaskHandler>()

    // ---- 生命周期 ----

    init {
        if (FeatureToggle.isEnabled(FeatureToggle.FeatureFlag.BATCH_GENERATION, context)) {
            Logger.i(TAG, "任务队列初始化")
            // 恢复未完成任务
            scope.launch {
                val running = taskRepo.getRunning()
                if (running != null) {
                    Logger.w(TAG, "发现未完成任务 ${running.id}，重新入队")
                    taskRepo.update(running.copy(status = "PENDING"))
                }
            }
        }
    }

    fun registerHandler(type: TaskType, handler: TaskHandler) {
        handlerRegistry[type] = handler
        Logger.d(TAG, "注册任务处理器: $type")
    }

    // ---- 公共 API ----

    fun enqueue(task: TaskEntity, priority: Int = 0) {
        scope.launch {
            val enriched = task.copy(priority = priority)
            taskRepo.enqueue(enriched)
            Logger.i(TAG, "入队 #${enriched.id} type=${enriched.type} pri=$priority")
            _taskLog.tryEmit(TaskLog("enqueue", "任务 #${enriched.id} 已加入队列"))
            if (_state.value == QueueState.IDLE) startProcessing()
        }
    }

    fun enqueueBatch(tasks: List<TaskEntity>) {
        tasks.forEachIndexed { i, t -> enqueue(t, priority = -i) }
    }

    fun pause() {
        isPaused = true
        _state.value = QueueState.PAUSED
        currentJob?.cancel()
        Logger.i(TAG, "队列已暂停")
        _taskLog.tryEmit(TaskLog("pause", "队列暂停"))
    }

    fun resume() {
        if (_state.value != QueueState.PAUSED) return
        isPaused = false
        _state.value = QueueState.RUNNING
        Logger.i(TAG, "队列已恢复")
        _taskLog.tryEmit(TaskLog("resume", "队列恢复"))
        startProcessing()
    }

    fun cancel(taskId: Long) {
        scope.launch {
            taskRepo.cancel(taskId)
            if (_currentTask.value?.id == taskId) {
                currentJob?.cancel()
                _currentTask.value = null
            }
            Logger.i(TAG, "取消任务 $taskId")
            _taskLog.tryEmit(TaskLog("cancel", "任务 $taskId 已取消"))
        }
    }

    fun retry(taskId: Long) {
        scope.launch {
            val task = taskRepo.getById(taskId)
            task?.let {
                taskRepo.update(it.copy(status = "PENDING", error = null))
                Logger.i(TAG, "重试任务 $taskId")
                if (_state.value == QueueState.IDLE) startProcessing()
            }
        }
    }

    fun clearFinished() {
        scope.launch {
            taskRepo.clearFinished()
            Logger.i(TAG, "已清理已完成任务")
        }
    }

    fun release() {
        scope.cancel()
        Logger.i(TAG, "任务队列已释放")
    }

    // ---- 内部 ----

    private fun startProcessing() {
        currentJob = scope.launch {
            _state.value = QueueState.RUNNING
            while (!isPaused) {
                val task = taskRepo.getPending().firstOrNull() ?: break
                _currentTask.value = task
                try {
                    taskRepo.update(task.copy(status = "RUNNING"))
                    executeTask(task)
                    taskRepo.update(task.copy(status = "DONE"))
                    Logger.i(TAG, "任务 ${task.id} 完成")
                    _taskLog.tryEmit(TaskLog("done", "任务 ${task.id} 完成"))
                } catch (ce: CancellationException) {
                    Logger.i(TAG, "任务 ${task.id} 取消")
                    break
                } catch (e: Exception) {
                    Logger.e(TAG, "任务 ${task.id} 失败", e)
                    taskRepo.update(task.copy(status = "FAILED", error = e.message))
                    _taskLog.tryEmit(TaskLog("error", "任务 ${task.id} 失败: ${e.message}"))
                    _state.value = QueueState.ERROR
                    // 错误后暂停，等待用户干预
                    isPaused = true
                    _state.value = QueueState.PAUSED
                    break
                }
            }
            if (!isPaused) {
                _state.value = QueueState.IDLE
                _currentTask.value = null
                _progress.value = 0f
            }
        }
    }

    private suspend fun executeTask(task: TaskEntity) {
        val handler = handlerRegistry[task.type] ?: run {
            Logger.w(TAG, "无处理器: ${task.type}，跳过")
            return
        }
        handler.execute(task) { p ->
            _progress.value = p
        }
    }
}

/**
 * 任务处理器接口 —— 新增任务类型只需实现此接口并注册
 */
interface TaskHandler {
    val type: TaskQueueManager.TaskType
    suspend fun execute(task: TaskEntity, onProgress: (Float) -> Unit)
}

data class TaskLog(val action: String, val message: String, val time: Long = System.currentTimeMillis())
