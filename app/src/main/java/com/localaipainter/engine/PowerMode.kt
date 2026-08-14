package com.localaipainter.engine

/**
 * 功率/性能模式枚举
 */
enum class PowerMode {
    /** 省电优先，降低频率/线程数 */
    POWER_SAVE,
    /** 性能与功耗均衡（默认） */
    BALANCED,
    /** 性能优先，允许高频运行 */
    PERFORMANCE,
    /** 极限性能，忽略温控 */
    EXTREME
}
