package com.localaipainter.engine

/**
 * 人脸修复模型类型
 */
enum class FaceRestoreType(val displayName: String, val nativeId: Int) {
    NONE      ("无",         -1),
    GFPGAN_V14("GFPGAN v1.4", 0),
    CODEFORMER("CodeFormer",  1),
    RESTOREFMR("RestoreFormer", 2),
    ;

    companion object {
        fun fromNative(id: Int): FaceRestoreType =
            values().find { it.nativeId == id } ?: NONE
    }
}
