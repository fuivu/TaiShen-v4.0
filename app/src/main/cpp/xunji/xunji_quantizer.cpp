/*
 * ═════════════════════════════════════════════════════════════
 *  迅集 (XunJi) — 量化核心实现  v4.0 TaiShen
 *  完整 7 种精度算法 (FP32/FP16/BF16/INT8/INT4/INT2/FP8)
 * ═════════════════════════════════════════════════════════════
 */
#include "xunji_quantizer.h"
#include <cstring>
#include <cfenv>
#include <cmath>

namespace xunji {

// ─── 工具 ─────────────────────────────────────────────────
const char* quant_level_name(QuantLevel l) {
    switch(l){case QuantLevel::FP32:return"fp32";case QuantLevel::FP16:return"fp16";
    case QuantLevel::BF16:return"bf16";case QuantLevel::INT8:return"int8";
    case QuantLevel::INT4:return"int4";case QuantLevel::INT2:return"int2";
    case QuantLevel::FP8:return"fp8";} return"unknown";
}
int quant_bits(QuantLevel l){switch(l){case QuantLevel::FP32:return 32;
    case QuantLevel::FP16:case QuantLevel::BF16:return 16;
    case QuantLevel::INT8:case QuantLevel::FP8:return 8;
    case QuantLevel::INT4:return 4;case QuantLevel::INT2:return 2;} return 32;
}
float quant_mem_ratio(QuantLevel l){switch(l){case QuantLevel::FP32:return 1.0f;
    case QuantLevel::FP16:case QuantLevel::BF16:return 0.5f;
    case QuantLevel::INT8:case QuantLevel::FP8:return 0.25f;
    case QuantLevel::INT4:return 0.125f;case QuantLevel::INT2:return 0.0625f;} return 1.0f;
}

static inline float clamp_f(float v,float a,float b){return v<a?a:v>b?b:v;}

// ─── FP16 ─────────────────────────────────────────────────
uint16_t fp32_to_fp16(float v){
    uint32_t x; std::memcpy(&x,&v,4);
    uint32_t sign=(x>>31)&1, exp=(x>>23)&0xFF, mant=x&0x7FFFFF;
    if(exp==0xFF){return(uint16_t)((sign<<15)|0x7C00|((mant)?1:0));}
    int e=(int)exp-127+15;
    if(e<=0){
        if(e<-10) return (uint16_t)(sign<<15);
        mant|=0x800000; uint16_t r=(uint16_t)((mant>>(14-e))+((mant>>(13-e))&1));
        return (uint16_t)((sign<<15)|r);
    }
    if(e>=31) return (uint16_t)((sign<<15)|0x7C00);
    return (uint16_t)((sign<<15)|(e<<10)|((mant>>13)&0x3FF));
}
float fp16_to_fp32(uint16_t h){
    uint16_t sign=(h>>15)&1, exp=(h>>10)&0x1F, mant=h&0x3FF;
    if(exp==0){
        if(mant==0) return sign?-0.0f:0.0f;
        int e=-14; while(!(mant&0x400)){mant<<=1;e--;}
        mant&=0x3FF; return std::ldexp((float)mant/1024.0f,e);
    }
    if(exp==31) return mant?std::numeric_limits<float>::quiet_NaN():(sign?-INFINITY:INFINITY);
    return std::ldexp((float)(mant|0x400)/1024.0f,(int)exp-15);
}
void quantize_fp16(const float* s,uint16_t* d,size_t n){for(size_t i=0;i<n;i++)d[i]=fp32_to_fp16(s[i]);}
void dequantize_fp16(const uint16_t* s,float* d,size_t n){for(size_t i=0;i<n;i++)d[i]=fp16_to_fp32(s[i]);}

// ─── BF16 ─────────────────────────────────────────────────
uint16_t fp32_to_bf16(float v){
    uint32_t x; std::memcpy(&x,&v,4);
    uint16_t r=(uint16_t)((x>>16)&0xFFFF);
    uint32_t lsb=(x>>16)&1; uint32_t halfway=(x&0x7FFF)>=0x8000;
    if(lsb&&halfway) r++;
    return r;
}
float bf16_to_fp32(uint16_t b){
    uint32_t x=(uint32_t)b<<16; float v; std::memcpy(&v,&x,4); return v;
}
void quantize_bf16(const float* s,uint16_t* d,size_t n){for(size_t i=0;i<n;i++)d[i]=fp32_to_bf16(s[i]);}
void dequantize_bf16(const uint16_t* s,float* d,size_t n){for(size_t i=0;i<n;i++)d[i]=bf16_to_fp32(s[i]);}

// ─── INT8 ────────────────────────────────────────────────
void compute_int8_params(const float* s,size_t n,Int8Params* p){
    float mn=s[0],mx=s[0]; for(size_t i=1;i<n;i++){if(s[i]<mn)mn=s[i];if(s[i]>mx)mx=s[i];}
    p->min_val=mn;p->max_val=mx;
    float range=mx-mn; if(range<=1e-8f){p->scale=1.0f;p->zero_point=0;}
    else{p->scale=range/255.0f;p->zero_point=(int)std::round(-mn/p->scale);}
}
void quantize_int8(const float* s,int8_t* d,size_t n,const Int8Params& p){
    for(size_t i=0;i<n;i++){float v=(s[i]-p.min_val)/p.scale+p.zero_point;
        int q=(int)std::round(clamp_f(v,-128.0f,127.0f)); d[i]=(int8_t)q;}
}
void dequantize_int8(const int8_t* s,float* d,size_t n,const Int8Params& p){
    for(size_t i=0;i<n;i++){d[i]=((float)s[i]-p.zero_point)*p.scale+p.min_val;}
}

// ─── INT4 ────────────────────────────────────────────────
void compute_int4_params(const float* s,size_t n,Int4Params* p){
    float mn=s[0],mx=s[0]; for(size_t i=1;i<n;i++){if(s[i]<mn)mn=s[i];if(s[i]>mx)mx=s[i];}
    float range=mx-mn; if(range<=1e-8f){p->scale=1.0f;p->zero_point=0;return;}
    p->scale=range/15.0f; p->zero_point=(int)std::round(-mn/p->scale);
}
void quantize_int4(const float* s,uint8_t* dst,size_t n,const Int4Params& p){
    std::memset(dst,0,(n+1)/2);
    float inv_scale = (p.scale > 1e-12f) ? (1.0f / p.scale) : 0.0f;
    for(size_t i=0;i<n;i++){
        float v = (s[i] - p.zero_point * p.scale) * inv_scale;
        int q = (int)std::round(clamp_f(v,-8.0f,7.0f));
        if(i%2==0) dst[i/2] = (uint8_t)(q & 0xF);
        else       dst[i/2]|= (uint8_t)((q & 0xF) << 4);
    }
}
void dequantize_int4(const uint8_t* src,float* d,size_t n,const Int4Params& p){
    for(size_t i=0;i<n;i++){
        int q = (i%2==0) ? (src[i/2] & 0xF) : ((src[i/2]>>4) & 0xF);
        if(q>=8) q-=16;
        d[i] = q * p.scale + p.zero_point * p.scale;
    }
}

// ─── INT2 ───────────────────────────────────────────────
void compute_int2_params(const float* s,size_t n,Int2Params* p){
    float mn=s[0],mx=s[0]; for(size_t i=1;i<n;i++){if(s[i]<mn)mn=s[i];if(s[i]>mx)mx=s[i];}
    float range=mx-mn; if(range<=1e-8f){p->scale=1.0f;p->zero_point=0;return;}
    p->scale=range/3.0f; p->zero_point=(int)std::round(-mn/p->scale);
}
void quantize_int2(const float* s,uint8_t* dst,size_t n,const Int2Params& p){
    std::memset(dst,0,(n+3)/4);
    float inv_scale = (p.scale > 1e-12f) ? (1.0f / p.scale) : 0.0f;
    for(size_t i=0;i<n;i++){
        float v = (s[i] - p.zero_point * p.scale) * inv_scale;
        int q = (int)std::round(clamp_f(v,-2.0f,1.0f));
        int slot = (i%4)*2;
        dst[i/4] |= ((uint8_t)(q & 3)) << slot;
    }
}
void dequantize_int2(const uint8_t* src,float* d,size_t n,const Int2Params& p){
    for(size_t i=0;i<n;i++){
        int q = (src[i/4] >> ((i%4)*2)) & 3;
        if(q>=2) q-=4;
        d[i] = q * p.scale + p.zero_point * p.scale;
    }
}

// ─── FP8 (E4M3) ─────────────────────────────────────────
uint8_t fp32_to_fp8_e4m3(float v){
    uint32_t x; std::memcpy(&x,&v,4);
    uint32_t sign=(x>>31)&1, exp=(x>>23)&0xFF, mant=x&0x7FFFFF;
    if(exp==0xFF){return(uint8_t)((sign<<7)|0x7F|((mant)?1:0));}
    int e=(int)exp-127+7;
    if(e<=0){
        if(e<-3) return (uint8_t)(sign<<7);
        mant|=0x800000; uint8_t r=(uint8_t)((mant>>(21-e))+((mant>>(20-e))&1));
        return (uint8_t)((sign<<7)|r);
    }
    if(e>=15) return (uint8_t)((sign<<7)|0x7F);
    return (uint8_t)((sign<<7)|(e<<3)|((mant>>20)&7));
}
float fp8_e4m3_to_fp32(uint8_t e){
    uint8_t sign=(e>>7)&1, exp=(e>>3)&0xF, mant=e&7;
    if(exp==0){
        if(mant==0) return sign?-0.0f:0.0f;
        return std::ldexp((float)mant/8.0f,(int)-6);
    }
    if(exp==15) return std::numeric_limits<float>::quiet_NaN();
    return std::ldexp((float)(mant|8)/8.0f,(int)exp-7);
}
void quantize_fp8(const float* s,uint8_t* d,size_t n){for(size_t i=0;i<n;i++)d[i]=fp32_to_fp8_e4m3(s[i]);}
void dequantize_fp8(const uint8_t* s,float* d,size_t n){for(size_t i=0;i<n;i++)d[i]=fp8_e4m3_to_fp32(s[i]);}

// ─── 统一分发 ──────────────────────────────────────────
bool quantize_uniform(const float* src,size_t n,QuantLevel lvl,QuantResult* out){
    if(!src||!out||n==0) return false;
    out->elem_count=n;
    switch(lvl){
        case QuantLevel::FP32:{
            out->type="fp32"; out->data.resize(n*4);
            std::memcpy(out->data.data(),src,n*4);
        }break;
        case QuantLevel::FP16:{
            out->type="fp16"; out->data.resize(n*2);
            quantize_fp16(src,(uint16_t*)out->data.data(),n);
        }break;
        case QuantLevel::BF16:{
            out->type="bf16"; out->data.resize(n*2);
            quantize_bf16(src,(uint16_t*)out->data.data(),n);
        }break;
        case QuantLevel::INT8:{
            out->type="int8"; out->data.resize(n);
            compute_int8_params(src,n,&out->int8_p);
            quantize_int8(src,(int8_t*)out->data.data(),n,out->int8_p);
        }break;
        case QuantLevel::INT4:{
            out->type="int4"; out->data.resize((n+1)/2);
            compute_int4_params(src,n,&out->int4_p);
            quantize_int4(src,out->data.data(),n,out->int4_p);
        }break;
        case QuantLevel::INT2:{
            out->type="int2"; out->data.resize((n+3)/4);
            compute_int2_params(src,n,&out->int2_p);
            quantize_int2(src,out->data.data(),n,out->int2_p);
        }break;
        case QuantLevel::FP8:{
            out->type="fp8"; out->data.resize(n);
            quantize_fp8(src,out->data.data(),n);
        }break;
        default:return false;
    }
    return true;
}

bool dequantize_uniform(const QuantResult& in,float* dst){
    if(!dst||in.elem_count==0) return false;
    if(in.type=="fp32"){std::memcpy(dst,in.data.data(),in.elem_count*4);}
    else if(in.type=="fp16") dequantize_fp16((const uint16_t*)in.data.data(),dst,in.elem_count);
    else if(in.type=="bf16") dequantize_bf16((const uint16_t*)in.data.data(),dst,in.elem_count);
    else if(in.type=="int8") dequantize_int8((const int8_t*)in.data.data(),dst,in.elem_count,in.int8_p);
    else if(in.type=="int4") dequantize_int4(in.data.data(),dst,in.elem_count,in.int4_p);
    else if(in.type=="int2") dequantize_int2(in.data.data(),dst,in.elem_count,in.int2_p);
    else if(in.type=="fp8")  dequantize_fp8(in.data.data(),dst,in.elem_count);
    else return false;
    return true;
}

} // namespace xunji
