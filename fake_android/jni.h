#ifndef _JNI_H_
#define _JNI_H_
#include <stdint.h>
typedef int32_t jint;
typedef int64_t jlong;
typedef float jfloat;
typedef double jdouble;
typedef char16_t jchar;
typedef int8_t jbyte;
typedef int16_t jshort;
typedef int64_t jlong;
typedef int32_t jboolean;
typedef void* jobject;
typedef void* jclass;
typedef void* jstring;
typedef void* jarray;
typedef void* jintArray;
typedef void* jfloatArray;
typedef void* jbyteArray;
typedef void* JNIEnv;
typedef void* JavaVM;
typedef void* jmethodID;
typedef void* jfieldID;
struct _jobject {};
#endif
