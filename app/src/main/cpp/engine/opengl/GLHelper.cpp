#include "GLHelper.h"
#include <android/log.h>
#include <string>
#include <vector>
#include <cstring>

#define TAG "GLHelper"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

namespace localaipainter {
namespace gl {

GLHelper::GLHelper() {
    LOGI("GLHelper created");
}

GLHelper::~GLHelper() {
    destroy();
}

bool GLHelper::init() {
    if (m_initialized) return true;

    LOGI("===== Initializing EGL + GLES3 =====");

    // 1. 获取 EGL Display
    m_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (m_display == EGL_NO_DISPLAY) {
        LOGE("eglGetDisplay failed");
        return false;
    }

    // 2. 初始化 EGL
    EGLint major, minor;
    if (!eglInitialize(m_display, &major, &minor)) {
        LOGE("eglInitialize failed (0x%x)", eglGetError());
        return false;
    }
    LOGI("EGL version: %d.%d", major, minor);

    // 3. 选择配置
    EGLint configAttribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_RED_SIZE,   8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE,  8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 0,
        EGL_STENCIL_SIZE, 0,
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_NONE
    };

    EGLConfig config = nullptr;
    EGLint numConfigs = 0;
    if (!eglChooseConfig(m_display, configAttribs, &config, 1, &numConfigs) || numConfigs == 0) {
        LOGE("eglChooseConfig failed");
        return false;
    }

    // 4. 创建 GLES3 上下文
    EGLint contextAttribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 3,
        EGL_NONE
    };
    m_context = eglCreateContext(m_display, config, EGL_NO_CONTEXT, contextAttribs);
    if (m_context == EGL_NO_CONTEXT) {
        LOGE("eglCreateContext failed (0x%x)", eglGetError());
        return false;
    }

    // 5. 创建 PBuffer Surface（离屏）
    EGLint surfaceAttribs[] = {
        EGL_WIDTH,  1,
        EGL_HEIGHT, 1,
        EGL_NONE
    };
    m_surface = eglCreatePbufferSurface(m_display, config, surfaceAttribs);
    if (m_surface == EGL_NO_SURFACE) {
        LOGE("eglCreatePbufferSurface failed");
        return false;
    }

    // 6. 绑定上下文
    if (!eglMakeCurrent(m_display, m_surface, m_surface, m_context)) {
        LOGE("eglMakeCurrent failed");
        return false;
    }

    // 7. 查询 GPU 信息
    queryGPUInfo();
    checkExtensions();

    m_initialized = true;
    LOGI("✅ EGL + GLES3 initialized");
    LOGI("   GPU: %s", m_gpuInfo.c_str());
    LOGI("   Compute Shader: %s", m_supportsCompute ? "YES" : "NO");
    return true;
}

void GLHelper::destroy() {
    if (!m_initialized) return;

    LOGI("Destroying GL resources...");

    if (m_display != EGL_NO_DISPLAY) {
        eglMakeCurrent(m_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (m_context != EGL_NO_CONTEXT) {
            eglDestroyContext(m_display, m_context);
        }
        if (m_surface != EGL_NO_SURFACE) {
            eglDestroySurface(m_display, m_surface);
        }
        eglTerminate(m_display);
    }

    m_display = EGL_NO_DISPLAY;
    m_context = EGL_NO_CONTEXT;
    m_surface = EGL_NO_SURFACE;
    m_initialized = false;
    m_textureMemoryMB = 0;
    LOGI("✅ GL resources destroyed");
}

void GLHelper::queryGPUInfo() {
    const char* vendor   = (const char*)glGetString(GL_VENDOR);
    const char* renderer = (const char*)glGetString(GL_RENDERER);
    const char* version  = (const char*)glGetString(GL_VERSION);
    const char* extensions = (const char*)glGetString(GL_EXTENSIONS);

    m_gpuInfo = std::string("Vendor=") + (vendor ? vendor : "?") +
                " | Renderer=" + (renderer ? renderer : "?") +
                " | Version=" + (version ? version : "?");

    LOGI("GPU Vendor: %s", vendor ? vendor : "?");
    LOGI("GPU Renderer: %s", renderer ? renderer : "?");
    LOGI("GL Version: %s", version ? version : "?");

    // 检查 Compute Shader 支持
    m_supportsCompute = (version && strstr(version, "OpenGL ES 3.1")) ||
                        (extensions && strstr(extensions, "GL_KHR_compute_shader"));
}

void GLHelper::checkExtensions() {
    const char* ext = (const char*)glGetString(GL_EXTENSIONS);
    if (!ext) return;

    if (strstr(ext, "GL_EXT_color_buffer_float")) {
        LOGD("✅ GL_EXT_color_buffer_float supported (RGBA32F textures)");
    } else {
        LOGD("⚠️ GL_EXT_color_buffer_float NOT supported (will use RGBA16F)");
    }

    if (strstr(ext, "GL_OES_texture_float_linear")) {
        LOGD("✅ GL_OES_texture_float_linear supported");
    }
}

GLuint GLHelper::createFloatTexture(int w, int h, const float* data) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    GLenum internalFormat = GL_RGBA32F;
    // 检查支持
    const char* ext = (const char*)glGetString(GL_EXTENSIONS);
    if (!ext || !strstr(ext, "GL_EXT_color_buffer_float")) {
        internalFormat = GL_RGBA16F; // Fallback to half-float
    }

    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, w, h, 0,
                  GL_RGBA, GL_FLOAT, data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    int bytesPerPixel = (internalFormat == GL_RGBA32F) ? 16 : 8;
    m_textureMemoryMB += (w * h * bytesPerPixel) / (1024 * 1024);

    LOGD("Created texture %u (%dx%d, %.1fMB)", tex, w, h,
          (w * h * bytesPerPixel) / (1024.0f * 1024.0f));
    return tex;
}

void GLHelper::uploadToTexture(GLuint tex, int w, int h, const float* data) {
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_FLOAT, data);
}

void GLHelper::readbackTexture(GLuint tex, int w, int h, float* outData) {
    glBindTexture(GL_TEXTURE_2D, tex);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, outData);
}

GLuint GLHelper::compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint status = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        GLchar info[1024];
        glGetShaderInfoLog(shader, 1024, nullptr, info);
        LOGE("Shader compile error: %s", info);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

GLuint GLHelper::createProgram(const char* vsSource, const char* fsSource) {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSource);
    if (!vs) return 0;
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSource);
    if (!fs) { glDeleteShader(vs); return 0; }

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    GLint status = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        GLchar info[1024];
        glGetProgramInfoLog(program, 1024, nullptr, info);
        LOGE("Program link error: %s", info);
        glDeleteProgram(program);
        program = 0;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

GLuint GLHelper::createFBO(GLuint colorTexture) {
    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_TEXTURE_2D, colorTexture, 0);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        LOGE("FBO incomplete: 0x%x", status);
        glDeleteFramebuffers(1, &fbo);
        return 0;
    }
    return fbo;
}

void GLHelper::deleteTexture(GLuint tex) {
    if (tex == 0) return;
    // 估算回收（简化）
    m_textureMemoryMB = (m_textureMemoryMB > 0) ? m_textureMemoryMB - 1 : 0;
    glDeleteTextures(1, &tex);
}

void GLHelper::deleteProgram(GLuint prog) {
    if (prog != 0) glDeleteProgram(prog);
}

int GLHelper::getMemoryUsageMB() const {
    return m_textureMemoryMB;
}

void GLHelper::drawFullscreen(GLuint program, int width, int height) {
    glUseProgram(program);
    glViewport(0, 0, width, height);

    // 全屏三角形（3个顶点，覆盖整个视口）
    GLfloat verts[] = {
        -1.0f, -1.0f,
         3.0f, -1.0f,
        -1.0f,  3.0f,
    };
    GLuint vao = 0, vbo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisableVertexAttribArray(0);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

} // namespace gl
} // namespace localaipainter
