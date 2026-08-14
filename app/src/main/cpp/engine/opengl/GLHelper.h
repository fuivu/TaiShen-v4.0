#pragma once

#include <GLES3/gl3.h>
#include <EGL/egl.h>
#include <android/native_window.h>
#include <cstdint>

namespace localaipainter {
namespace gl {

/**
 * GLHelper —— C++ 侧 OpenGL ES 辅助工具
 *
 * 提供：
 *   - EGL 上下文快速创建/销毁
 *   - 离屏渲染表面管理
 *   - 着色器编译/链接工具函数
 *   - 纹理创建/上传/下载
 *   - FBO 管理
 *   - Compute Shader 检测（GLES 3.1+）
 *
 * Kotlin 侧的 OpenGLEngine 调用这些 JNI 方法
 */
class GLHelper {
public:
    GLHelper();
    ~GLHelper();

    // 禁止拷贝
    GLHelper(const GLHelper&) = delete;
    GLHelper& operator=(const GLHelper&) = delete;

    /**
     * 初始化 EGL + GLES3 上下文（离屏 PBuffer）
     * @return true 成功
     */
    bool init();

    /**
     * 销毁所有 GL 资源
     */
    void destroy();

    /**
     * 获取 GPU 信息字符串
     */
    const char* getGPUInfo() const { return m_gpuInfo.c_str(); }

    /**
     * 检查是否支持 Compute Shader（GLES >= 3.1）
     */
    bool supportsComputeShader() const { return m_supportsCompute; }

    /**
     * 创建 RGBA32F 纹理
     * @param w 宽度
     * @param h 高度
     * @param data 初始数据（nullptr = 空）
     * @return GLuint texture ID, 0 = 失败
     */
    GLuint createFloatTexture(int w, int h, const float* data = nullptr);

    /**
     * 上传数据到纹理
     */
    void uploadToTexture(GLuint tex, int w, int h, const float* data);

    /**
     * 从纹理读回数据
     */
    void readbackTexture(GLuint tex, int w, int h, float* outData);

    /**
     * 创建并编译着色器
     */
    GLuint compileShader(GLenum type, const char* source);

    /**
     * 创建并链接程序
     */
    GLuint createProgram(const char* vsSource, const char* fsSource);

    /**
     * 创建 FBO + 绑定纹理
     */
    GLuint createFBO(GLuint colorTexture);

    /**
     * 删除纹理
     */
    void deleteTexture(GLuint tex);

    /**
     * 删除程序
     */
    void deleteProgram(GLuint prog);

    /**
     * 获取当前显存使用估算（MB）
     */
    int getMemoryUsageMB() const;

    /**
     * 执行全屏绘制（用指定程序）
     */
    void drawFullscreen(GLuint program, int width, int height);

    bool isInitialized() const { return m_initialized; }

private:
    // EGL
    EGLDisplay  m_display = EGL_NO_DISPLAY;
    EGLContext  m_context = EGL_NO_CONTEXT;
    EGLSurface  m_surface = EGL_NO_SURFACE;

    // 状态
    bool        m_initialized = false;
    bool        m_supportsCompute = false;
    std::string m_gpuInfo;

    // 内存跟踪
    int         m_textureMemoryMB = 0;

    // 内部
    void queryGPUInfo();
    void checkExtensions();
};

} // namespace gl
} // namespace localaipainter
