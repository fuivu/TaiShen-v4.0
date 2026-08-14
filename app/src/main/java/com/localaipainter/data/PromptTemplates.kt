package com.localaipainter.data

/**
 * 内置提示词模板库 — 18 个高质量模板
 * 覆盖人像/风景/动漫/奇幻/科幻/美食/动物/建筑/抽象/Logo/摄影等分类
 */
data class PromptTemplate(
    val id: String,
    val name: String,
    val category: String,
    val prompt: String,
    val negativePrompt: String = "ugly, deformed, blurry, low quality, jpeg artifacts, watermark, text, signature",
    val thumbnailEmoji: String = "🎨",
)

object PromptTemplates {

    val ALL: List<PromptTemplate> = listOf(

        // ===== 人像 =====
        PromptTemplate(
            id = "portrait_01", name = "电影人像", category = "人像",
            prompt = "cinematic portrait of a woman, soft natural light, shallow depth of field, 85mm lens, film grain, kodak portra 400, high detail skin texture, professional photography",
            negativePrompt = "ugly, deformed, blurry, low quality, extra fingers, bad anatomy, watermark",
            thumbnailEmoji = "🎬"
        ),
        PromptTemplate(
            id = "portrait_02", name = "赛博朋克", category = "人像",
            prompt = "cyberpunk portrait, neon lights, reflective sunglasses, chrome implants, dark atmosphere, volumetric fog, rim light, hyper detailed, 8k uhd",
            thumbnailEmoji = "🤖"
        ),
        PromptTemplate(
            id = "portrait_03", name = "古典油画", category = "人像",
            prompt = "renaissance oil painting portrait, baroque lighting, rich colors, masterwork, classical composition, intricate details, canvas texture, rembrandt lighting",
            thumbnailEmoji = "🖼️"
        ),

        // ===== 风景 =====
        PromptTemplate(
            id = "landscape_01", name = "山川日落", category = "风景",
            prompt = "majestic mountain landscape at sunset, golden hour, dramatic clouds, reflections on lake, pine forest, volumetric light rays, ultra detailed, national geographic photography",
            thumbnailEmoji = "🏔️"
        ),
        PromptTemplate(
            id = "landscape_02", name = "梦幻森林", category = "风景",
            prompt = "enchanted forest, glowing mushrooms, fairy lights, misty atmosphere, ancient trees, magical creatures, fantasy illustration, highly detailed, lush greenery",
            thumbnailEmoji = "🌲"
        ),
        PromptTemplate(
            id = "landscape_03", name = "城市夜景", category = "风景",
            prompt = "tokyo street at night, neon signs, rain reflections, cyberpunk cityscape, bokeh lights, cinematic wide angle, moody atmosphere, hyper realistic",
            thumbnailEmoji = "🌃"
        ),

        // ===== 动漫 =====
        PromptTemplate(
            id = "anime_01", name = "日系少女", category = "动漫",
            prompt = "anime girl, school uniform, cherry blossom petals, soft pastel colors, big sparkling eyes, detailed hair, studio ghibli style, beautiful background, masterpiece",
            thumbnailEmoji = "🌸"
        ),
        PromptTemplate(
            id = "anime_02", name = "机甲战斗", category = "动漫",
            prompt = "mecha battle scene, giant robot, explosions, dynamic action pose, sci-fi battlefield, detailed mechanical parts, dramatic lighting, anime style, epic composition",
            thumbnailEmoji = "🚀"
        ),

        // ===== 奇幻 =====
        PromptTemplate(
            id = "fantasy_01", name = "龙与魔法", category = "奇幻",
            prompt = "ancient dragon breathing fire, medieval castle in background, dark stormy sky, magical energy particles, epic fantasy art, intricate scales, dramatic lighting, 8k resolution",
            thumbnailEmoji = "🐉"
        ),
        PromptTemplate(
            id = "fantasy_02", name = "精灵森林", category = "奇幻",
            prompt = "elf princess in mystical forest, glowing runes, ethereal light, intricate armor, flowing silver hair, fantasy illustration, highly detailed, magical atmosphere",
            thumbnailEmoji = "🧝"
        ),

        // ===== 科幻 =====
        PromptTemplate(
            id = "scifi_01", name = "太空站", category = "科幻",
            prompt = "futuristic space station orbiting alien planet, massive windows, advanced technology, holographic displays, sci-fi concept art, cinematic lighting, ultra detailed, 8k",
            thumbnailEmoji = "🛸"
        ),
        PromptTemplate(
            id = "scifi_02", name = "AI 核心", category = "科幻",
            prompt = "artificial intelligence core, glowing neural network, data streams, futuristic server room, blue and purple neon, technological singularity, hyper detailed, dramatic perspective",
            thumbnailEmoji = "🧠"
        ),

        // ===== 美食 =====
        PromptTemplate(
            id = "food_01", name = "精致料理", category = "美食",
            prompt = "gourmet sushi platter, fresh ingredients, wooden table, soft natural lighting, food photography, shallow depth of field, appetizing, professional styling, 4k",
            thumbnailEmoji = "🍣"
        ),

        // ===== 动物 =====
        PromptTemplate(
            id = "animal_01", name = "雪山之狼", category = "动物",
            prompt = "majestic white wolf on snowy mountain peak, piercing blue eyes, dramatic winter storm, fur details, national geographic style, cinematic lighting, ultra sharp",
            thumbnailEmoji = "🐺"
        ),

        // ===== 建筑 =====
        PromptTemplate(
            id = "arch_01", name = "未来建筑", category = "建筑",
            prompt = "futuristic architecture, curved glass facade, floating building, sustainable design, green walls, sunset reflection, architectural visualization, hyper detailed, 8k render",
            thumbnailEmoji = "🏙️"
        ),

        // ===== 抽象 =====
        PromptTemplate(
            id = "abstract_01", name = "流体艺术", category = "抽象",
            prompt = "fluid art abstract painting, flowing colors, acrylic pour, vibrant red gold blue, marble texture, mesmerizing patterns, fine art photography, high contrast",
            thumbnailEmoji = "🎭"
        ),

        // ===== Logo =====
        PromptTemplate(
            id = "logo_01", name = "科技 Logo", category = "Logo",
            prompt = "minimalist tech company logo, geometric shapes, gradient blue to purple, clean lines, vector style, professional branding, white background, modern design",
            negativePrompt = "text, words, letters, watermark, complex, cluttered, blurry",
            thumbnailEmoji = "💎"
        ),

        // ===== 摄影 =====
        PromptTemplate(
            id = "photo_01", name = "街头摄影", category = "摄影",
            prompt = "street photography, candid moment, black and white, grainy film, 35mm lens, urban decay, dramatic shadows, decisive moment, documentary style, raw emotion",
            thumbnailEmoji = "📷"
        ),
    )

    fun byCategory(category: String): List<PromptTemplate> =
        ALL.filter { it.category == category }

    fun categories(): List<String> =
        ALL.map { it.category }.distinct()

    fun getById(id: String): PromptTemplate? =
        ALL.find { it.id == id }
}
