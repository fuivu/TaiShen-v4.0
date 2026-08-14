package com.localaipainter.engine

/**
 * 采样器类型 — 对应 C++ SchedulerType
 */
enum class SchedulerType(val displayName: String, val nativeName: String) {
    EULER_A   ("Euler A",     "euler_a"),
    EULER     ("Euler",       "euler"),
    DDIM      ("DDIM",        "ddim"),
    DPM_2     ("DPM-2",      "dpm_2"),
    DPM_2_A   ("DPM-2 A",    "dpm_2_a"),
    DPMPP_2S  ("DPM++ 2S a", "dpmpp_2s"),
    DPMPP_2M  ("DPM++ 2M",   "dpmpp_2m"),
    DPMPP_SDE ("DPM++ SDE",  "dpmpp_sde"),
    HEUN      ("Heun",        "heun"),
    LCM       ("LCM",         "lcm"),
    UNI_PC    ("UniPC",      "uni_pc"),
    DEIS      ("DEIS",        "deis"),
    ;

    companion object {
        fun fromNative(name: String): SchedulerType {
            return values().find { it.nativeName.equals(name, ignoreCase = true) }
                ?: EULER_A
        }
        fun fromDisplayName(name: String): SchedulerType {
            return values().find { it.displayName.equals(name, ignoreCase = true) }
                ?: EULER_A
        }
    }
}
