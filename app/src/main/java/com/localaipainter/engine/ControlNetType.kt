package com.localaipainter.engine

/**
 * ControlNet 类型
 */
enum class ControlNetType(val displayName: String, val nativeId: Int) {
    NONE      ("无",        -1),
    CANNY     ("Canny 边缘", 0),
    OPENPOSE  ("OpenPose",  1),
    DEPTH     ("深度图",    2),
    SCRIBBLE  ("涂鸦",      3),
    MLSD      ("线段检测",   4),
    SEG       ("语义分割",   5),
    ;

    companion object {
        fun fromNative(id: Int): ControlNetType =
            values().find { it.nativeId == id } ?: NONE
    }
}
