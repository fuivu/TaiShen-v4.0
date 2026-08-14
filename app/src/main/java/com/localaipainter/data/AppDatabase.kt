package com.localaipainter.data

import android.content.Context
import androidx.room.*
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.localaipainter.data.dao.*
import com.localaipainter.data.entity.*

/**
 * Room 数据库 —— 统一入口（TaiShen v4.0）
 *
 * 实体类定义位于 data/entity/ 目录：
 *   - ModelEntity       (models)
 *   - HistoryEntity     (history)
 *   - GenerationConfig  (generation_configs)
 *   - LoRAEntity       (lora_models)
 *   - TaskEntity       (tasks)
 *   - GenerationHistory(generation_history)
 *
 * DAO 接口位于 data/dao/ 目录。
 *
 * 版本历史：
 *   v1 → v2 : 新增 ModelEntity 4 列 + LoRA/Task 表 + GenConfig 字段
 *   v2 → v3 : 新增 GenerationHistory + ModelEntity 扩展字段
 *   v3 → v4 : 统一 entity 包路径，消除重复定义
 */
@Database(
    entities = [
        ModelEntity::class,
        HistoryEntity::class,
        GenerationConfig::class,
        LoRAEntity::class,
        TaskEntity::class,
        GenerationHistory::class,
    ],
    version = 4,
    exportSchema = false,
)
@TypeConverters(Converters::class)
abstract class AppDatabase : RoomDatabase() {

    // ─── DAOs（来自 data/dao/ 目录）──────────────
    abstract fun modelDao(): ModelDao
    abstract fun historyDao(): HistoryDao
    abstract fun generationConfigDao(): GenerationConfigDao
    abstract fun loraDao(): LoRADao
    abstract fun taskDao(): TaskDao
    abstract fun generationHistoryDao(): GenerationHistoryDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        /**
         * v1 → v2 迁移
         */
        private val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE models ADD COLUMN isFavorite INTEGER NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE models ADD COLUMN lastUsedAt INTEGER NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE models ADD COLUMN description TEXT NOT NULL DEFAULT ''")
                db.execSQL("ALTER TABLE models ADD COLUMN thumbnailPath TEXT NOT NULL DEFAULT ''")
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS `lora_models` (
                        `id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        `name` TEXT NOT NULL,
                        `path` TEXT NOT NULL,
                        `type` TEXT NOT NULL DEFAULT 'lora',
                        `rank` INTEGER NOT NULL DEFAULT 4,
                        `alpha` REAL NOT NULL DEFAULT 1.0,
                        `enabled` INTEGER NOT NULL DEFAULT 1,
                        `size_mb` REAL NOT NULL DEFAULT 0,
                        `imported_at` INTEGER NOT NULL DEFAULT 0,
                        `trigger_words` TEXT NOT NULL DEFAULT ''
                    )
                """)
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS `tasks` (
                        `id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        `name` TEXT NOT NULL,
                        `type` TEXT NOT NULL,
                        `status` TEXT NOT NULL DEFAULT 'pending',
                        `progress` REAL NOT NULL DEFAULT 0,
                        `params_json` TEXT NOT NULL DEFAULT '{}',
                        `result_path` TEXT,
                        `created_at` INTEGER NOT NULL DEFAULT 0,
                        `updated_at` INTEGER NOT NULL DEFAULT 0,
                        `error_msg` TEXT
                    )
                """)
                db.execSQL("ALTER TABLE `generation_configs` ADD COLUMN `sampler` TEXT NOT NULL DEFAULT 'euler_a'")
                db.execSQL("ALTER TABLE `generation_configs` ADD COLUMN `seed` INTEGER NOT NULL DEFAULT -1")
                db.execSQL("ALTER TABLE `generation_configs` ADD COLUMN `controlnet_type` TEXT NOT NULL DEFAULT ''")
            }
        }

        /**
         * v2 → v3 迁移
         */
        private val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS `generation_history` (
                        `id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        `prompt` TEXT NOT NULL,
                        `negative_prompt` TEXT NOT NULL DEFAULT '',
                        `width` INTEGER NOT NULL DEFAULT 512,
                        `height` INTEGER NOT NULL DEFAULT 512,
                        `steps` INTEGER NOT NULL DEFAULT 20,
                        `cfg_scale` REAL NOT NULL DEFAULT 7.5,
                        `seed` INTEGER NOT NULL DEFAULT -1,
                        `sampler` TEXT NOT NULL DEFAULT 'euler_a',
                        `model_name` TEXT NOT NULL DEFAULT '',
                        `lora_names` TEXT NOT NULL DEFAULT '',
                        `image_path` TEXT,
                        `thumb_path` TEXT,
                        `latency_ms` REAL NOT NULL DEFAULT 0,
                        `peak_memory_mb` REAL NOT NULL DEFAULT 0,
                        `backend_used` TEXT NOT NULL DEFAULT '',
                        `created_at` INTEGER NOT NULL DEFAULT 0,
                        `favorite` INTEGER NOT NULL DEFAULT 0,
                        `tags` TEXT NOT NULL DEFAULT ''
                    )
                """)
                db.execSQL("ALTER TABLE `models` ADD COLUMN `format` TEXT NOT NULL DEFAULT 'safetensors'")
                db.execSQL("ALTER TABLE `models` ADD COLUMN `quantization` TEXT NOT NULL DEFAULT 'fp16'")
                db.execSQL("ALTER TABLE `models` ADD COLUMN `architecture` TEXT NOT NULL DEFAULT 'sd15'")
                db.execSQL("ALTER TABLE `models` ADD COLUMN `supports_int8` INTEGER NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE `models` ADD COLUMN `supports_int4` INTEGER NOT NULL DEFAULT 0")
            }
        }

        /**
         * v3 → v4 迁移：重命名列以对齐 entity/ 定义
         *   imported_at → importedAt
         *   size_mb → sizeMb
         */
        private val MIGRATION_3_4 = object : Migration(3, 4) {
            override fun migrate(db: SupportSQLiteDatabase) {
                // ModelEntity: 列名已一致（在 v1→v2 中用 snake_case 创建）
                // 这里做最终对齐，确保与新 entity 类匹配
                db.execSQL("ALTER TABLE `models` ADD COLUMN `backend` TEXT NOT NULL DEFAULT ''")
                db.execSQL("ALTER TABLE `models` ADD COLUMN `precision` TEXT NOT NULL DEFAULT 'fp16'")
                db.execSQL("ALTER TABLE `models` ADD COLUMN `fileSize` INTEGER NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE `models` ADD COLUMN `verified` INTEGER NOT NULL DEFAULT 0")
            }
        }

        fun getInstance(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "localaipainter.db",
                )
                    .addMigrations(MIGRATION_1_2, MIGRATION_2_3, MIGRATION_3_4)
                    .fallbackToDestructiveMigration()
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
//  TypeConverters
// ═══════════════════════════════════════════════════════

class Converters {
    @TypeConverter
    fun fromStringList(value: List<String>): String =
        value.joinToString("||")

    @TypeConverter
    fun toStringList(value: String): List<String> =
        if (value.isEmpty()) emptyList() else value.split("||")

    @TypeConverter
    fun fromLong(value: Long): Long = value

    @TypeConverter
    fun toLong(value: Long): Long = value

    @TypeConverter
    fun fromInt(value: Int): Int = value

    @TypeConverter
    fun toInt(value: Int): Int = value

    @TypeConverter
    fun fromBoolean(value: Boolean): Int = if (value) 1 else 0

    @TypeConverter
    fun toBoolean(value: Int): Boolean = value != 0
}
